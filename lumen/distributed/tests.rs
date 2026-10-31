//! Unit tests for `DeviceMesh`, mesh backends, TPU slice env, the TCP store and the provided collectives.
use super::DeviceMesh;

/// Converts string slices to owned dimension names.
fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| n.to_string()).collect()
}

/// Builds a 2x4 cuda mesh named `(dp, tp)` over ranks 0..8 as seen from `rank`.
fn mesh_2x4(rank: usize) -> DeviceMesh {
    DeviceMesh::new(
        "cuda",
        vec![2, 4],
        (0..8).collect(),
        Some(names(&["dp", "tp"])),
        rank,
        8,
    )
    .unwrap()
}

/// Coordinates and local ranks follow row-major order over the mesh shape.
#[test]
fn coordinates_are_row_major() {
    assert_eq!(mesh_2x4(0).coordinate(), Some(vec![0, 0]));
    assert_eq!(mesh_2x4(5).coordinate(), Some(vec![1, 1]));
    assert_eq!(mesh_2x4(7).local_rank(1), Ok(3));
}

/// A process whose rank isn't in the mesh has no coordinate or local rank.
#[test]
fn a_rank_outside_the_mesh_has_no_coordinate() {
    let mesh = DeviceMesh::new("cpu", vec![2], vec![0, 1], None, 3, 4).unwrap();
    assert_eq!(mesh.coordinate(), None);
    assert!(mesh.local_rank(0).is_err());
}

/// Slicing one dimension keeps the row or column that contains the calling rank.
#[test]
fn slicing_keeps_the_submesh_of_the_rank() {
    let dp = mesh_2x4(5).slice(&names(&["dp"])).unwrap();
    assert_eq!((dp.shape(), dp.ranks()), (&[2][..], &[1, 5][..]));
    let tp = mesh_2x4(5).slice(&names(&["tp"])).unwrap();
    assert_eq!(tp.ranks(), &[4, 5, 6, 7]);
    assert_eq!(tp.coordinate(), Some(vec![1]));
}

/// Slicing with the dimension names reversed yields the transposed mesh.
#[test]
fn slicing_in_another_order_transposes() {
    let mesh = mesh_2x4(2).slice(&names(&["tp", "dp"])).unwrap();
    assert_eq!(mesh.shape(), &[4, 2]);
    assert_eq!(mesh.ranks(), &[0, 4, 1, 5, 2, 6, 3, 7]);
    assert_eq!(mesh.coordinate(), Some(vec![2, 0]));
}

/// Slicing indexes through the mesh's rank list rather than assuming ranks equal positions.
#[test]
fn slicing_follows_an_arbitrary_rank_map() {
    let mesh = DeviceMesh::new(
        "tpu",
        vec![2, 2],
        vec![3, 1, 2, 0],
        Some(names(&["a", "b"])),
        2,
        4,
    )
    .unwrap();

    assert_eq!(mesh.slice(&names(&["a"])).unwrap().ranks(), &[3, 2]);
    assert_eq!(mesh.slice(&names(&["b"])).unwrap().ranks(), &[2, 0]);
}

/// Construction and slicing reject bad device types, shapes, duplicate ranks and bad dimension names.
#[test]
fn invalid_meshes_are_rejected() {
    let make = |device_type, shape: Vec<usize>, ranks: Vec<usize>, dims: Option<&[&str]>| {
        DeviceMesh::new(device_type, shape, ranks, dims.map(names), 0, 4).unwrap_err()
    };

    assert!(make("gpu", vec![2], vec![0, 1], None).contains("device_type"));
    assert!(make("cpu", vec![3], vec![0, 1], None).contains("holds 3 ranks"));
    assert!(make("cpu", vec![2], vec![1, 1], None).contains("more than once"));
    assert!(make("cpu", vec![2], vec![0, 4], None).contains("outside the world"));
    assert!(make("cpu", vec![2], vec![0, 1], Some(&["a", "b"])).contains("each of"));
    assert!(make("cpu", vec![1, 2], vec![0, 1], Some(&["a", "a"])).contains("unique"));
    assert!(mesh_2x4(0).slice(&names(&["pp"])).is_err());
    assert!(mesh_2x4(0).slice(&names(&["dp", "dp"])).is_err());
}

/// An MPS mesh is the single device in a single process.
#[test]
fn an_mps_mesh_is_the_one_device_in_one_process() {
    let mps = super::mesh_backend("mps").unwrap();

    if !mps.is_available() {
        assert!(mps.world(1).is_err());
        return;
    }

    assert_eq!(mps.world(1), Ok((0, 1)));
    let err = mps.world(2).unwrap_err();
    assert!(err.contains("the 1 MPS device, got 2"), "{err}");
}

/// Every device type name resolves to a mesh backend of that name.
#[test]
fn every_device_type_has_a_mesh_backend() {
    for name in ["cpu", "mps", "cuda", "tpu", "meta"] {
        assert_eq!(super::mesh_backend(name).unwrap().name(), name);
    }

    assert!(
        super::mesh_backend("gpu")
            .err()
            .unwrap()
            .contains("device_type")
    );
}

/// The TPU slice env pins the process to its chip, picks its slicebuilder port and enables insecure gRPC once.
#[test]
fn a_tpu_process_joins_its_slice_with_its_chip() {
    let addresses = "localhost:9000,localhost:9001,localhost:9002,localhost:9003";
    let env = super::backend::tpu_slice_env("2,2,1", addresses, 6, 2, 8, "--flag=1").unwrap_err();
    assert!(env.contains("several TPU slices"), "{env}");

    let env: std::collections::HashMap<_, _> =
        super::backend::tpu_slice_env("2,2,1", addresses, 2, 2, 4, "--flag=1")
            .unwrap()
            .into_iter()
            .collect();

    assert_eq!(env["CLOUD_TPU_TASK_ID"], "2");
    assert_eq!(env["TPU_VISIBLE_CHIPS"], "2");
    assert_eq!(env["TPU_VISIBLE_DEVICES"], "2");
    assert_eq!(env["TPU_PROCESS_BOUNDS"], "2,2,1");
    assert_eq!(env["TPU_HOST_BOUNDS"], "2,2,1");
    assert_eq!(env["TPU_CHIPS_PER_PROCESS_BOUNDS"], "1,1,1");
    assert_eq!(env["TPU_CHIPS_PER_HOST_BOUNDS"], "1,1,1");
    assert_eq!(env["TPU_PROCESS_PORT"], "9002");
    assert_eq!(env["TPU_PROCESS_ADDRESSES"], addresses);

    assert_eq!(
        env["LIBTPU_INIT_ARGS"],
        "--flag=1 --slicebuilder_use_insecure_grpc=true"
    );

    let four_d = super::backend::tpu_slice_env(
        "2,2,1,2",
        addresses,
        0,
        0,
        4,
        "--slicebuilder_use_insecure_grpc=false",
    )
    .unwrap();

    assert!(four_d.contains(&("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1,1".into())));

    assert!(four_d.contains(&(
        "LIBTPU_INIT_ARGS",
        "--slicebuilder_use_insecure_grpc=false".into()
    )));
}

/// Tests of the TCP store server and client over loopback.
mod store {
    use std::thread;
    use std::time::Duration;

    use super::super::TcpStore;

    /// Generous timeout so slow CI machines don't flake.
    const TIMEOUT: Duration = Duration::from_secs(10);

    /// Starts a master store on a free loopback port.
    fn master() -> TcpStore {
        TcpStore::new("127.0.0.1", 0, true, TIMEOUT).unwrap()
    }

    /// Connects a non-master client to `master`.
    fn client(master: &TcpStore) -> TcpStore {
        TcpStore::new("127.0.0.1", master.port(), false, TIMEOUT).unwrap()
    }

    /// Values set by any client are visible to the others.
    #[test]
    #[cfg_attr(miri, ignore = "sockets, which Miri isolates")]
    fn clients_share_the_masters_values() {
        let master = master();
        let client = client(&master);
        master.set("key", b"value").unwrap();
        assert_eq!(client.get("key").unwrap(), b"value");
        client.set("key", b"").unwrap();
        assert_eq!(master.get("key").unwrap(), b"");
    }

    /// `delete_key` removes a key for every client and reports whether it existed.
    #[test]
    #[cfg_attr(miri, ignore = "sockets, which Miri isolates")]
    fn delete_key_removes_the_key() {
        let master = master();
        let client = client(&master);
        master.set("key", b"value").unwrap();
        assert_eq!(client.delete_key("key"), Ok(true));
        assert_eq!(master.delete_key("key"), Ok(false));
        let err = client
            .wait(&["key"], Duration::from_millis(20))
            .unwrap_err();
        assert!(err.contains("timed out"), "{err}");
    }

    /// `get` blocks until another client sets the key.
    #[test]
    #[cfg_attr(miri, ignore = "sockets, which Miri isolates")]
    fn get_waits_for_the_key() {
        let master = master();
        let port = master.port();

        let setter = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));

            TcpStore::new("127.0.0.1", port, false, TIMEOUT)
                .unwrap()
                .set("late", b"here")
                .unwrap();
        });

        assert_eq!(master.get("late").unwrap(), b"here");
        setter.join().unwrap();
    }

    /// Concurrent `add`s are atomic, each seeing a distinct running total.
    #[test]
    #[cfg_attr(miri, ignore = "sockets, which Miri isolates")]
    fn add_counts_across_clients() {
        let master = master();
        let port = master.port();

        let adders: Vec<_> = (0..8)
            .map(|_| {
                thread::spawn(move || {
                    TcpStore::new("127.0.0.1", port, false, TIMEOUT)
                        .unwrap()
                        .add("count", 2)
                        .unwrap()
                })
            })
            .collect();

        let mut seen: Vec<i64> = adders.into_iter().map(|a| a.join().unwrap()).collect();
        seen.sort();
        assert_eq!(seen, (1..=8).map(|i| 2 * i).collect::<Vec<_>>());
        assert_eq!(master.get("count").unwrap(), b"16");
        assert_eq!(master.add("count", -16).unwrap(), 0);
    }

    /// `get` and `wait` fail with a timeout error naming the missing key.
    #[test]
    #[cfg_attr(miri, ignore = "sockets, which Miri isolates")]
    fn waiting_too_long_is_an_error() {
        let master = TcpStore::new("127.0.0.1", 0, true, Duration::from_millis(50)).unwrap();
        assert!(master.get("never").unwrap_err().contains("timed out"));
        master.set("a", b"").unwrap();

        let err = master
            .wait(&["a", "b"], Duration::from_millis(50))
            .unwrap_err();

        assert!(err.contains("\"b\""), "{err}");
        master.wait(&["a"], Duration::from_millis(50)).unwrap();
    }

    /// Connecting to a port nobody listens on fails once the timeout passes.
    #[test]
    #[cfg_attr(miri, ignore = "sockets, which Miri isolates")]
    fn connecting_to_no_store_times_out() {
        // bind and immediately drop a listener to get a port that is free but closed
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();

        let err = TcpStore::new("127.0.0.1", port, false, Duration::from_millis(50)).err();
        assert!(err.unwrap().contains("connecting to the store"));
    }
}

/// Tests of the collectives `Communicator` provides from sends and receives, over threads.
mod collective {
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;

    use super::super::collective::Communicator;
    use crate::DType;
    use crate::graph::Reduction;

    /// Per (sender, receiver) position, the messages in flight in the order sent.
    type Queues = HashMap<(usize, usize), VecDeque<Vec<u8>>>;

    /// Messages in flight between the threads of a group.
    #[derive(Default)]
    struct Mailbox {
        queues: Mutex<Queues>,
        posted: Condvar,
    }

    /// A process of a group whose processes are threads sharing a `Mailbox`; sends never wait.
    struct Thread {
        rank: usize,
        size: usize,
        mailbox: Arc<Mailbox>,
    }

    impl Communicator for Thread {
        /// This thread's position.
        fn rank(&self) -> usize {
            self.rank
        }

        /// The number of threads.
        fn size(&self) -> usize {
            self.size
        }

        /// Queues the bytes for `peer`.
        unsafe fn send(
            &self,
            send: *const u8,
            count: usize,
            dtype: DType,
            peer: usize,
        ) -> Result<(), String> {
            let bytes =
                unsafe { std::slice::from_raw_parts(send, count * dtype.size_of()) }.to_vec();
            let mut queues = self.mailbox.queues.lock().unwrap();
            queues
                .entry((self.rank, peer))
                .or_default()
                .push_back(bytes);
            self.mailbox.posted.notify_all();
            Ok(())
        }

        /// Waits for the next message from `peer` and copies it into `recv`.
        unsafe fn recv(
            &self,
            recv: *mut u8,
            count: usize,
            dtype: DType,
            peer: usize,
        ) -> Result<(), String> {
            let mut queues = self.mailbox.queues.lock().unwrap();
            loop {
                if let Some(bytes) = queues
                    .get_mut(&(peer, self.rank))
                    .and_then(VecDeque::pop_front)
                {
                    assert_eq!(bytes.len(), count * dtype.size_of());
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), recv, bytes.len()) };
                    return Ok(());
                }
                queues = self.mailbox.posted.wait(queues).unwrap();
            }
        }

        /// Writes zero bytes.
        unsafe fn zero(&self, recv: *mut u8, count: usize, dtype: DType) -> Result<(), String> {
            unsafe { std::ptr::write_bytes(recv, 0, count * dtype.size_of()) };
            Ok(())
        }

        /// Not exercised: only the provided collectives are tested.
        unsafe fn all_reduce(
            &self,
            _: *const u8,
            _: *mut u8,
            _: usize,
            _: DType,
            _: Reduction,
        ) -> Result<(), String> {
            unreachable!()
        }

        /// Not exercised: only the provided collectives are tested.
        unsafe fn reduce(
            &self,
            _: *const u8,
            _: *mut u8,
            _: usize,
            _: DType,
            _: Reduction,
            _: usize,
        ) -> Result<(), String> {
            unreachable!()
        }

        /// Not exercised: only the provided collectives are tested.
        unsafe fn reduce_scatter(
            &self,
            _: *const u8,
            _: *mut u8,
            _: usize,
            _: DType,
            _: Reduction,
        ) -> Result<(), String> {
            unreachable!()
        }
    }

    /// Runs `body` on a thread per process of a group of `size`, each holding the values `10 * rank + i` for
    /// `i` in `0..6`, and returns each one's 6 result values.
    fn on_threads(size: usize, body: fn(&Thread, &[i32], &mut [i32])) -> Vec<Vec<i32>> {
        let mailbox = Arc::new(Mailbox::default());
        let threads: Vec<_> = (0..size)
            .map(|rank| {
                let mailbox = mailbox.clone();
                thread::spawn(move || {
                    let comm = Thread {
                        rank,
                        size,
                        mailbox,
                    };
                    let send: Vec<i32> = (0..6).map(|i| 10 * rank as i32 + i).collect();
                    let mut recv = vec![-1; 6];
                    body(&comm, &send, &mut recv);
                    recv
                })
            })
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).collect()
    }

    /// Pointers to `send` and `recv` as the communicator takes them.
    fn raw(send: &[i32], recv: &mut [i32]) -> (*const u8, *mut u8) {
        (send.as_ptr().cast(), recv.as_mut_ptr().cast())
    }

    /// Gathers, all-to-alls, broadcasts and scatters built from sends and receives place each block where
    /// its definition says, over three processes of 2-element blocks.
    #[test]
    fn provided_collectives_move_blocks_where_they_belong() {
        let gathered = on_threads(3, |c, s, r| unsafe {
            let (s, r) = raw(s, r);
            c.all_gather(s, r, 2, DType::I32).unwrap()
        });
        assert!(gathered.iter().all(|r| r == &[0, 1, 10, 11, 20, 21]));
        let exchanged = on_threads(3, |c, s, r| unsafe {
            let (s, r) = raw(s, r);
            c.all_to_all(s, r, 2, DType::I32).unwrap()
        });
        assert_eq!(exchanged[1], [2, 3, 12, 13, 22, 23]);
        let broadcast = on_threads(3, |c, s, r| unsafe {
            let (s, r) = raw(s, r);
            c.broadcast(s, r, 6, DType::I32, 2).unwrap()
        });
        assert!(broadcast.iter().all(|r| r == &[20, 21, 22, 23, 24, 25]));
        let gathered = on_threads(3, |c, s, r| unsafe {
            let (s, r) = raw(s, r);
            c.gather(s, r, 2, DType::I32, 1).unwrap()
        });
        assert_eq!(
            gathered,
            [vec![0; 6], vec![0, 1, 10, 11, 20, 21], vec![0; 6]]
        );
        let scattered = on_threads(3, |c, s, r| unsafe {
            let (s, r) = raw(s, r);
            c.scatter(s, r, 2, DType::I32, 0).unwrap()
        });
        assert_eq!(
            scattered.iter().map(|r| &r[..2]).collect::<Vec<_>>(),
            [[0, 1], [2, 3], [4, 5]]
        );
    }
}
