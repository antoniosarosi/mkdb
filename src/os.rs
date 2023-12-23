//! OS specific code.

use std::{io, path::Path};

/// IO device block size.
pub(crate) trait HardwareBlockSize {
    /// Returns the underlying IO device block size or preferred IO buffer size.
    fn block_size(&self) -> io::Result<usize>;
}

/// Used to implement [`HardwareBlockSize`] on different platforms.
pub(crate) struct Disk<'p> {
    /// A valid path in the disk file system.
    path: &'p Path,
}

impl<'p, P: AsRef<Path>> From<&'p P> for Disk<'p> {
    fn from(path: &'p P) -> Self {
        Self {
            path: path.as_ref(),
        }
    }
}

#[cfg(unix)]
mod unix {
    use std::{fs::File, io, os::unix::prelude::MetadataExt};

    use super::{Disk, HardwareBlockSize};

    impl<'p> HardwareBlockSize for Disk<'p> {
        fn block_size(&self) -> io::Result<usize> {
            File::open(self.path)
                .and_then(|file| file.metadata())
                .map(|metadata| metadata.blksize() as usize)
        }
    }
}

#[cfg(windows)]
mod windows {
    use std::{io, os::windows::ffi::OsStrExt};

    use windows::{
        core::PCWSTR,
        Win32::{Foundation::MAX_PATH, Storage::FileSystem},
    };

    use super::{Disk, HardwareBlockSize};

    impl<'p> HardwareBlockSize for Disk<'p> {
        fn block_size(&self) -> io::Result<usize> {
            unsafe {
                let mut volume = [0u16; MAX_PATH as usize];

                let mut windows_file_path =
                    self.path.as_os_str().encode_wide().collect::<Vec<u16>>();

                // encode_wide() does not add the null terminator.
                windows_file_path.push(0);

                FileSystem::GetVolumePathNameW(
                    PCWSTR::from_raw(windows_file_path.as_ptr()),
                    &mut volume,
                )?;

                let mut bytes_per_sector: u32 = 0;
                let mut sectors_per_cluster: u32 = 0;

                FileSystem::GetDiskFreeSpaceW(
                    PCWSTR::from_raw(volume.as_ptr()),
                    Some(&mut bytes_per_sector),
                    Some(&mut sectors_per_cluster),
                    None,
                    None,
                )?;

                Ok((bytes_per_sector * sectors_per_cluster) as usize)
            }
        }
    }
}
