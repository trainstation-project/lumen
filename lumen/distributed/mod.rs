pub mod backend;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod store;
#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::Device;
pub use backend::{MeshBackend, mesh_backend};
pub use store::TcpStore;

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30 * 60);

const JOINED: &str = "init_process_group/joined";
const READY: &str = "init_process_group/ready";

struct World {
    rank: usize,
    world_size: usize,
    store: Arc<TcpStore>,
}

static WORLD: Mutex<Option<World>> = Mutex::new(None);

fn world() -> std::sync::MutexGuard<'static, Option<World>> {
    WORLD.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn env_index(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Err(_) => Ok(default),
        Ok(value) => value
            .parse()
            .map_err(|_| format!("{name} must be a non-negative integer, got '{value}'")),
    }
}

fn required_env(name: &str) -> Result<String, String> {
    std::env::var(name).map_err(|_| format!("env:// rendezvous: {name} is not set"))
}

fn required_env_index(name: &str) -> Result<usize, String> {
    required_env(name)?;
    env_index(name, 0)
}

pub fn get_rank() -> Result<usize, String> {
    match &*world() {
        Some(world) => Ok(world.rank),
        None => env_index("RANK", 0),
    }
}

pub fn get_world_size() -> Result<usize, String> {
    match &*world() {
        Some(world) => Ok(world.world_size),
        None => env_index("WORLD_SIZE", 1),
    }
}

pub fn is_initialized() -> bool {
    world().is_some()
}

pub fn default_store() -> Option<Arc<TcpStore>> {
    world().as_ref().map(|w| w.store.clone())
}

pub fn rendezvous(
    init_method: &str,
    rank: Option<usize>,
    world_size: Option<usize>,
    timeout: Duration,
) -> Result<(Arc<TcpStore>, usize, usize), String> {
    let (host, port, rank, world_size) = match init_method {
        "env://" => {
            let rank = rank.map_or_else(|| required_env_index("RANK"), Ok)?;
            let world_size = world_size.map_or_else(|| required_env_index("WORLD_SIZE"), Ok)?;
            let port = required_env("MASTER_PORT")?;
            (required_env("MASTER_ADDR")?, port, rank, world_size)
        }
        url => {
            let address = url.strip_prefix("tcp://").ok_or_else(|| {
                format!("init_method must be env:// or tcp://host:port, got '{url}'")
            })?;
            let (host, port) = address
                .rsplit_once(':')
                .ok_or_else(|| format!("tcp:// rendezvous: '{url}' has no port"))?;
            let missing = |name| format!("tcp:// rendezvous needs {name}");
            (
                host.to_owned(),
                port.to_owned(),
                rank.ok_or_else(|| missing("rank"))?,
                world_size.ok_or_else(|| missing("world_size"))?,
            )
        }
    };
    let port = port
        .parse()
        .map_err(|_| format!("the store's port must be a number, got '{port}'"))?;
    let store = TcpStore::new(&host, port, rank == 0, timeout)?;
    Ok((Arc::new(store), rank, world_size))
}

pub fn init_process_group(
    store: Arc<TcpStore>,
    rank: usize,
    world_size: usize,
) -> Result<(), String> {
    if is_initialized() {
        return Err("the default process group is already initialized".into());
    }
    if rank >= world_size {
        return Err(format!(
            "rank {rank} is outside the world of size {world_size}"
        ));
    }
    if store.add(JOINED, 1)? == world_size as i64 {
        store.set(READY, b"")?;
    }
    store.wait(&[READY], store.timeout())?;
    *world() = Some(World {
        rank,
        world_size,
        store,
    });
    Ok(())
}

pub fn destroy_process_group() -> Result<(), String> {
    world()
        .take()
        .map(drop)
        .ok_or_else(|| "the default process group is not initialized".into())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceMesh {
    device_type: &'static str,
    shape: Vec<usize>,
    ranks: Vec<usize>,
    dim_names: Option<Vec<String>>,
    rank: usize,
}

impl DeviceMesh {
    pub fn new(
        device_type: &str,
        shape: Vec<usize>,
        ranks: Vec<usize>,
        dim_names: Option<Vec<String>>,
        rank: usize,
        world_size: usize,
    ) -> Result<Self, String> {
        let device_type = Device::of_type(device_type, 0)
            .ok_or_else(|| {
                format!("device_type must be one of cpu, mps, cuda, tpu, meta, got '{device_type}'")
            })?
            .backend()
            .name();
        if shape.is_empty() {
            return Err("a DeviceMesh needs at least one dimension".into());
        }
        if shape.iter().product::<usize>() != ranks.len() {
            return Err(format!(
                "a mesh of shape {shape:?} holds {} ranks, got {}",
                shape.iter().product::<usize>(),
                ranks.len()
            ));
        }
        let mut seen = HashSet::new();
        for &r in &ranks {
            if !seen.insert(r) {
                return Err(format!("rank {r} appears in the mesh more than once"));
            }
            if r >= world_size {
                return Err(format!(
                    "mesh rank {r} is outside the world of size {world_size} (WORLD_SIZE)"
                ));
            }
        }
        if let Some(names) = &dim_names {
            if names.len() != shape.len() {
                return Err(format!(
                    "mesh_dim_names {names:?} must name each of the mesh's {} dimensions",
                    shape.len()
                ));
            }
            if names.iter().collect::<HashSet<_>>().len() != names.len() {
                return Err(format!("mesh_dim_names {names:?} must be unique"));
            }
        }
        Ok(DeviceMesh {
            device_type,
            shape,
            ranks,
            dim_names,
            rank,
        })
    }

    pub fn device_type(&self) -> &'static str {
        self.device_type
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ranks(&self) -> &[usize] {
        &self.ranks
    }

    pub fn dim_names(&self) -> Option<&[String]> {
        self.dim_names.as_deref()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    pub fn dim(&self, name: &str) -> Result<usize, String> {
        let names = self.dim_names.as_ref().ok_or_else(|| {
            format!("cannot find mesh dimension '{name}': the mesh has no mesh_dim_names")
        })?;
        names.iter().position(|n| n == name).ok_or_else(|| {
            format!("mesh dimension '{name}' does not exist; the mesh's are {names:?}")
        })
    }

    fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.ndim()];
        for d in (0..self.ndim() - 1).rev() {
            strides[d] = strides[d + 1] * self.shape[d + 1];
        }
        strides
    }

    pub fn coordinate(&self) -> Option<Vec<usize>> {
        let flat = self.ranks.iter().position(|&r| r == self.rank)?;
        Some(
            self.strides()
                .iter()
                .zip(&self.shape)
                .map(|(stride, size)| flat / stride % size)
                .collect(),
        )
    }

    pub fn local_rank(&self, dim: usize) -> Result<usize, String> {
        let coordinate = self
            .coordinate()
            .ok_or_else(|| format!("rank {} is not in the mesh", self.rank))?;
        Ok(coordinate[dim])
    }

    pub fn slice(&self, names: &[String]) -> Result<DeviceMesh, String> {
        let dims = names
            .iter()
            .map(|n| self.dim(n))
            .collect::<Result<Vec<_>, _>>()?;
        if dims.iter().collect::<HashSet<_>>().len() != dims.len() {
            return Err(format!("mesh dimensions {names:?} must be unique"));
        }
        let coordinate = self.coordinate().ok_or_else(|| {
            format!(
                "rank {} is not in the mesh: it has no submesh to slice",
                self.rank
            )
        })?;
        let strides = self.strides();
        let base: usize = (0..self.ndim())
            .filter(|d| !dims.contains(d))
            .map(|d| coordinate[d] * strides[d])
            .sum();
        let shape: Vec<usize> = dims.iter().map(|&d| self.shape[d]).collect();
        let count: usize = shape.iter().product();
        let ranks = (0..count)
            .map(|i| {
                let mut rest = i;
                let mut flat = base;
                for (&d, &size) in dims.iter().zip(&shape).rev() {
                    flat += rest % size * strides[d];
                    rest /= size;
                }
                self.ranks[flat]
            })
            .collect();
        Ok(DeviceMesh {
            device_type: self.device_type,
            shape,
            ranks,
            dim_names: Some(names.to_vec()),
            rank: self.rank,
        })
    }
}
