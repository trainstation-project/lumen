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
