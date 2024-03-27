//! OS specific code.

use std::{fs::File, io, path::Path};

/// Filesystem operations.
pub(crate) trait Fs {
    /// Returns the underlying IO device block size or preferred IO buffer size.
    fn disk_block_size(file: impl AsRef<Path>) -> io::Result<usize>;

    /// Locks the file so that only one existing process can use it.
    fn lock(file: &File) -> io::Result<()>;
}

/// Used to implement [`Fs`] on different operating systems.
pub(crate) struct Os;

#[cfg(unix)]
mod unix {
    use std::{
        fs::File,
        io,
        os::{fd::AsRawFd, unix::prelude::MetadataExt},
        path::Path,
    };

    use super::{Fs, Os};

    impl Fs for Os {
        fn disk_block_size(path: impl AsRef<Path>) -> io::Result<usize> {
            Ok(File::open(&path)?.metadata()?.blksize() as usize)
        }

        fn lock(file: &File) -> io::Result<()> {
            let lock = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };

            if lock != 0 {
                return Err(io::Error::new(io::ErrorKind::Other, "could not lock file"));
            }

            Ok(())
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

    impl<'p> Fs for Disk<'p> {
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
