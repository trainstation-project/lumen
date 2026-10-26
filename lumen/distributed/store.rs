//! A minimal TCP key-value store for rendezvous: a threaded server on the master rank and a blocking client.
//!
//! Wire format: a one-byte opcode, then little-endian `u64` lengths/values and length-prefixed byte strings.
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

/// Request opcode: store a value.
const SET: u8 = 0;
/// Request opcode: read a value, waiting until it exists.
const GET: u8 = 1;
/// Request opcode: add to an integer counter.
const ADD: u8 = 2;
/// Request opcode: wait until a set of keys exists.
const WAIT: u8 = 3;

/// Reply status: the request succeeded.
const FOUND: u8 = 0;
/// Reply status: the awaited keys did not appear before the timeout.
const TIMED_OUT: u8 = 1;

/// Pause between attempts while the server isn't accepting connections yet.
const CONNECT_RETRY: Duration = Duration::from_millis(10);

/// The server's key to bytes map.
type Values = HashMap<String, Vec<u8>>;

/// Server-side table whose condvar wakes waiters whenever a key changes.
#[derive(Default)]
struct Table {
    values: Mutex<Values>,
    changed: Condvar,
}

impl Table {
    /// Locks the map, recovering it if a holder panicked.
    fn lock(&self) -> MutexGuard<'_, Values> {
        self.values.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Blocks until all `keys` exist and returns the locked map, or `None` once `timeout` elapses.
    fn wait_for(&self, keys: &[String], timeout: Duration) -> Option<MutexGuard<'_, Values>> {
        let deadline = Instant::now() + timeout;
        let mut values = self.lock();
        while !keys.iter().all(|k| values.contains_key(k)) {
            let left = deadline.checked_duration_since(Instant::now())?;
            values = self
                .changed
                .wait_timeout(values, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Some(values)
    }

    /// Stores `value` under `key` and wakes waiters.
    fn set(&self, key: String, value: Vec<u8>) {
        self.lock().insert(key, value);
        self.changed.notify_all();
    }

    /// Adds `amount` to the decimal integer at `key` (0 if absent or unparsable) and returns the sum.
    fn add(&self, key: String, amount: i64) -> i64 {
        let mut values = self.lock();
        let current = values
            .get(&key)
            .and_then(|v| std::str::from_utf8(v).ok()?.parse::<i64>().ok())
            .unwrap_or(0);
        let sum = current + amount;
        values.insert(key, sum.to_string().into_bytes());
        drop(values);
        self.changed.notify_all();
        sum
    }
}

/// Writes `value` as 8 little-endian bytes.
fn write_u64(stream: &mut TcpStream, value: u64) -> std::io::Result<()> {
    stream.write_all(&value.to_le_bytes())
}

/// Reads 8 little-endian bytes as a `u64`.
fn read_u64(stream: &mut TcpStream) -> std::io::Result<u64> {
    let mut bytes = [0; 8];
    stream.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Writes a single byte.
fn write_u8(stream: &mut TcpStream, value: u8) -> std::io::Result<()> {
    stream.write_all(&[value])
}

/// Reads a single byte.
fn read_u8(stream: &mut TcpStream) -> std::io::Result<u8> {
    let mut byte = [0];
    stream.read_exact(&mut byte)?;
    Ok(byte[0])
}

/// Writes `bytes` prefixed by their length.
fn write_bytes(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    write_u64(stream, bytes.len() as u64)?;
    stream.write_all(bytes)
}

/// Reads a length-prefixed byte string.
fn read_bytes(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut bytes = vec![0; read_u64(stream)? as usize];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Reads a length-prefixed UTF-8 key.
fn read_key(stream: &mut TcpStream) -> std::io::Result<String> {
    String::from_utf8(read_bytes(stream)?)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Reads a timeout sent as whole milliseconds.
fn read_timeout(stream: &mut TcpStream) -> std::io::Result<Duration> {
    Ok(Duration::from_millis(read_u64(stream)?))
}

/// Reads one request from `stream`, serves it against `table` and writes the reply.
fn answer(stream: &mut TcpStream, table: &Table) -> std::io::Result<()> {
    match read_u8(stream)? {
        SET => {
            let key = read_key(stream)?;
            table.set(key, read_bytes(stream)?);
            write_u8(stream, FOUND)
        }
        GET => {
            let key = read_key(stream)?;
            let timeout = read_timeout(stream)?;
            let value = table
                .wait_for(std::slice::from_ref(&key), timeout)
                .map(|values| values[&key].clone());
            match value {
                Some(value) => {
                    write_u8(stream, FOUND)?;
                    write_bytes(stream, &value)
                }
                None => write_u8(stream, TIMED_OUT),
            }
        }
        ADD => {
            let key = read_key(stream)?;
            let amount = read_u64(stream)? as i64;
            write_u64(stream, table.add(key, amount) as u64)
        }
        WAIT => {
            let keys = (0..read_u64(stream)?)
                .map(|_| read_key(stream))
                .collect::<std::io::Result<Vec<_>>>()?;
            let timeout = read_timeout(stream)?;
            let found = table.wait_for(&keys, timeout).is_some();
            write_u8(stream, if found { FOUND } else { TIMED_OUT })
        }
        op => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown store request {op}"),
        )),
    }
}

/// Background store server; dropping it stops the accept loop.
struct Server {
    address: SocketAddr,
    stopped: Arc<AtomicBool>,
}

impl Server {
    /// Listens on `port` (0 picks a free one) and serves each connection on its own thread.
    fn start(port: u16) -> Result<Self, String> {
        let listener = TcpListener::bind(("0.0.0.0", port))
            .map_err(|e| format!("the store cannot listen on port {port}: {e}"))?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let stopped = Arc::new(AtomicBool::new(false));
        let table = Arc::new(Table::default());
        let stop = stopped.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                let Ok(mut stream) = stream else { continue };
                let table = table.clone();
                thread::spawn(move || while answer(&mut stream, &table).is_ok() {});
            }
        });
        Ok(Server { address, stopped })
    }
}

impl Drop for Server {
    /// Stops the accept loop.
    fn drop(&mut self) {
        // wake the blocking accept with a throwaway connection so the loop sees `stopped`
        self.stopped.store(true, Ordering::Release);
        let _ = TcpStream::connect(("127.0.0.1", self.address.port()));
    }
}

/// Client connection to a store, also owning the server when this process is the master.
pub struct TcpStore {
    host: String,
    port: u16,
    /// Default wait for `get` and the process group barrier.
    timeout: Duration,
    stream: Mutex<TcpStream>,
    /// Keeps the server alive for as long as the master's client.
    _server: Option<Server>,
}

impl TcpStore {
    /// Starts the server when `is_master`, then connects to `host:port`, retrying until `timeout`.
    pub fn new(host: &str, port: u16, is_master: bool, timeout: Duration) -> Result<Self, String> {
        let server = is_master.then(|| Server::start(port)).transpose()?;
        let port = server.as_ref().map_or(port, |s| s.address.port());
        let stream = Self::connect(host, port, timeout)?;
        Ok(TcpStore {
            host: host.to_owned(),
            port,
            timeout,
            stream: Mutex::new(stream),
            _server: server,
        })
    }

    /// Connects to `host:port`, retrying every `CONNECT_RETRY` until `timeout` so workers may start before the master.
    fn connect(host: &str, port: u16, timeout: Duration) -> Result<TcpStream, String> {
        let deadline = Instant::now() + timeout;
        loop {
            let error = match (host, port).to_socket_addrs() {
                Err(e) => e,
                Ok(addresses) => {
                    let mut error = None;
                    for address in addresses {
                        match TcpStream::connect(address) {
                            Ok(stream) => {
                                let _ = stream.set_nodelay(true);
                                return Ok(stream);
                            }
                            Err(e) => error = Some(e),
                        }
                    }
                    error.unwrap_or_else(|| std::io::ErrorKind::AddrNotAvailable.into())
                }
            };
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out after {:?} connecting to the store at {host}:{port}: {error}",
                    timeout
                ));
            }
            thread::sleep(CONNECT_RETRY);
        }
    }

    /// Returns the host this client connected to.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Returns the store's port, the bound one when this process picked port 0.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Returns the default timeout for blocking reads.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Runs one request/reply exchange under the connection lock.
    fn request<T>(
        &self,
        send: impl FnOnce(&mut TcpStream) -> std::io::Result<T>,
    ) -> Result<T, String> {
        let mut stream = self.stream.lock().unwrap_or_else(|e| e.into_inner());
        send(&mut stream).map_err(|e| format!("lost the store at {}:{}: {e}", self.host, self.port))
    }

    /// Formats the timeout error for `keys`.
    fn timed_out(&self, keys: &[&str], timeout: Duration) -> String {
        format!("timed out after {timeout:?} waiting for {keys:?} in the store")
    }

    /// Stores `value` under `key`.
    pub fn set(&self, key: &str, value: &[u8]) -> Result<(), String> {
        self.request(|s| {
            write_u8(s, SET)?;
            write_bytes(s, key.as_bytes())?;
            write_bytes(s, value)?;
            read_u8(s).map(drop)
        })
    }

    /// Returns the value under `key`, waiting up to the store's timeout for it to be set.
    pub fn get(&self, key: &str) -> Result<Vec<u8>, String> {
        let timeout = self.timeout;
        let value = self.request(|s| {
            write_u8(s, GET)?;
            write_bytes(s, key.as_bytes())?;
            write_u64(s, timeout.as_millis() as u64)?;
            match read_u8(s)? {
                FOUND => read_bytes(s).map(Some),
                _ => Ok(None),
            }
        })?;
        value.ok_or_else(|| self.timed_out(&[key], timeout))
    }

    /// Atomically adds `amount` to the counter at `key` and returns the new value.
    pub fn add(&self, key: &str, amount: i64) -> Result<i64, String> {
        self.request(|s| {
            write_u8(s, ADD)?;
            write_bytes(s, key.as_bytes())?;
            write_u64(s, amount as u64)?;
            read_u64(s).map(|sum| sum as i64)
        })
    }

    /// Blocks until every key in `keys` exists or `timeout` elapses.
    pub fn wait(&self, keys: &[&str], timeout: Duration) -> Result<(), String> {
        let found = self.request(|s| {
            write_u8(s, WAIT)?;
            write_u64(s, keys.len() as u64)?;
            for key in keys {
                write_bytes(s, key.as_bytes())?;
            }
            write_u64(s, timeout.as_millis() as u64)?;
            Ok(read_u8(s)? == FOUND)
        })?;
        match found {
            true => Ok(()),
            false => Err(self.timed_out(keys, timeout)),
        }
    }
}
