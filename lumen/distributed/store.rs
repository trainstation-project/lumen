use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

const SET: u8 = 0;
const GET: u8 = 1;
const ADD: u8 = 2;
const WAIT: u8 = 3;

const FOUND: u8 = 0;
const TIMED_OUT: u8 = 1;

const CONNECT_RETRY: Duration = Duration::from_millis(10);

type Values = HashMap<String, Vec<u8>>;

#[derive(Default)]
struct Table {
    values: Mutex<Values>,
    changed: Condvar,
}

impl Table {
    fn lock(&self) -> MutexGuard<'_, Values> {
        self.values.lock().unwrap_or_else(|e| e.into_inner())
    }

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

    fn set(&self, key: String, value: Vec<u8>) {
        self.lock().insert(key, value);
        self.changed.notify_all();
    }

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

fn write_u64(stream: &mut TcpStream, value: u64) -> std::io::Result<()> {
    stream.write_all(&value.to_le_bytes())
}

fn read_u64(stream: &mut TcpStream) -> std::io::Result<u64> {
    let mut bytes = [0; 8];
    stream.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_u8(stream: &mut TcpStream, value: u8) -> std::io::Result<()> {
    stream.write_all(&[value])
}

fn read_u8(stream: &mut TcpStream) -> std::io::Result<u8> {
    let mut byte = [0];
    stream.read_exact(&mut byte)?;
    Ok(byte[0])
}

fn write_bytes(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    write_u64(stream, bytes.len() as u64)?;
    stream.write_all(bytes)
}

fn read_bytes(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut bytes = vec![0; read_u64(stream)? as usize];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_key(stream: &mut TcpStream) -> std::io::Result<String> {
    String::from_utf8(read_bytes(stream)?)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn read_timeout(stream: &mut TcpStream) -> std::io::Result<Duration> {
    Ok(Duration::from_millis(read_u64(stream)?))
}

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

struct Server {
    address: SocketAddr,
    stopped: Arc<AtomicBool>,
}

impl Server {
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
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        let _ = TcpStream::connect(("127.0.0.1", self.address.port()));
    }
}

pub struct TcpStore {
    host: String,
    port: u16,
    timeout: Duration,
    stream: Mutex<TcpStream>,
    _server: Option<Server>,
}

impl TcpStore {
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

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    fn request<T>(
        &self,
        send: impl FnOnce(&mut TcpStream) -> std::io::Result<T>,
    ) -> Result<T, String> {
        let mut stream = self.stream.lock().unwrap_or_else(|e| e.into_inner());
        send(&mut stream).map_err(|e| format!("lost the store at {}:{}: {e}", self.host, self.port))
    }

    fn timed_out(&self, keys: &[&str], timeout: Duration) -> String {
        format!("timed out after {timeout:?} waiting for {keys:?} in the store")
    }

    pub fn set(&self, key: &str, value: &[u8]) -> Result<(), String> {
        self.request(|s| {
            write_u8(s, SET)?;
            write_bytes(s, key.as_bytes())?;
            write_bytes(s, value)?;
            read_u8(s).map(drop)
        })
    }

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

    pub fn add(&self, key: &str, amount: i64) -> Result<i64, String> {
        self.request(|s| {
            write_u8(s, ADD)?;
            write_bytes(s, key.as_bytes())?;
            write_u64(s, amount as u64)?;
            read_u64(s).map(|sum| sum as i64)
        })
    }

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
