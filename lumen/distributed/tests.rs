use super::DeviceMesh;

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| n.to_string()).collect()
}

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

#[test]
fn coordinates_are_row_major() {
    assert_eq!(mesh_2x4(0).coordinate(), Some(vec![0, 0]));
    assert_eq!(mesh_2x4(5).coordinate(), Some(vec![1, 1]));
    assert_eq!(mesh_2x4(7).local_rank(1), Ok(3));
}

#[test]
fn a_rank_outside_the_mesh_has_no_coordinate() {
    let mesh = DeviceMesh::new("cpu", vec![2], vec![0, 1], None, 3, 4).unwrap();
    assert_eq!(mesh.coordinate(), None);
    assert!(mesh.local_rank(0).is_err());
}

#[test]
fn slicing_keeps_the_submesh_of_the_rank() {
    let dp = mesh_2x4(5).slice(&names(&["dp"])).unwrap();
    assert_eq!((dp.shape(), dp.ranks()), (&[2][..], &[1, 5][..]));
    let tp = mesh_2x4(5).slice(&names(&["tp"])).unwrap();
    assert_eq!(tp.ranks(), &[4, 5, 6, 7]);
    assert_eq!(tp.coordinate(), Some(vec![1]));
}

#[test]
fn slicing_in_another_order_transposes() {
    let mesh = mesh_2x4(2).slice(&names(&["tp", "dp"])).unwrap();
    assert_eq!(mesh.shape(), &[4, 2]);
    assert_eq!(mesh.ranks(), &[0, 4, 1, 5, 2, 6, 3, 7]);
    assert_eq!(mesh.coordinate(), Some(vec![2, 0]));
}

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

mod store {
    use std::thread;
    use std::time::Duration;

    use super::super::TcpStore;

    const TIMEOUT: Duration = Duration::from_secs(10);

    fn master() -> TcpStore {
        TcpStore::new("127.0.0.1", 0, true, TIMEOUT).unwrap()
    }

    fn client(master: &TcpStore) -> TcpStore {
        TcpStore::new("127.0.0.1", master.port(), false, TIMEOUT).unwrap()
    }

    #[test]
    fn clients_share_the_masters_values() {
        let master = master();
        let client = client(&master);
        master.set("key", b"value").unwrap();
        assert_eq!(client.get("key").unwrap(), b"value");
        client.set("key", b"").unwrap();
        assert_eq!(master.get("key").unwrap(), b"");
    }

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
    fn connecting_to_no_store_times_out() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let err = TcpStore::new("127.0.0.1", port, false, Duration::from_millis(50)).err();
        assert!(err.unwrap().contains("connecting to the store"));
    }
}
