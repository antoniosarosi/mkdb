//! OS specific code.

use std::{
    fs::{self, File},
    io,
    path::Path,
};

/// Necessary syscalls to get the storage hardware block size.
pub(crate) trait DiskBlockSize {
    /// Returns the underlying IO device block size or preferred IO buffer size.
    fn disk_block_size(path: impl AsRef<Path>) -> io::Result<usize>;
}

/// Trait for opening a file depending on the OS.
pub(crate) trait Open {
    /// Opens the file with the specified [`OpenOptions`].
    fn open(self, path: impl AsRef<Path>) -> io::Result<File>;
}

/// Used to implement [`DiskBlockSize`] and [`Open`] traits on different
/// operating systems.
pub(crate) struct Fs;

impl Fs {
    /// See [`OpenOptions`].
    pub fn options() -> OpenOptions {
        OpenOptions {
            inner: File::options(),
            bypass_cache: false,
            lock: false,
        }
    }
}

/// Just like [`fs::OpenOptions`] but with some additional options that are
/// handled differently depending on the operating system.
pub(crate) struct OpenOptions {
    /// Inner [`std::fs::OpenOptions`] instance.
    inner: fs::OpenOptions,
    /// Bypasses OS cache.
    bypass_cache: bool,
    /// Locks the file exclusively for the calling process.
    lock: bool,
}

impl OpenOptions {
    /// Disables the OS cache for reads and writes.
    pub fn bypass_cache(mut self, bypass_cache: bool) -> Self {
        self.bypass_cache = bypass_cache;
        self
    }

    /// Locks the file with exclusive access for the calling process.
    ///
    /// File won't actually be locked until [`Self::open`] is called.
    pub fn lock(mut self, lock: bool) -> Self {
        self.lock = lock;
        self
    }

    /// Create if doesn't exist.
    pub fn create(mut self, create: bool) -> Self {
        self.inner.create(create);
        self
    }

    /// Open for reading.
    pub fn read(mut self, read: bool) -> Self {
        self.inner.read(read);
        self
    }

    /// Open for writing.
    pub fn write(mut self, write: bool) -> Self {
        self.inner.write(write);
        self
    }

    /// Set the length of the file to 0 bytes if it exists.
    pub fn truncate(mut self, truncate: bool) -> Self {
        self.inner.truncate(truncate);
        self
    }
}

#[cfg(unix)]
mod unix {
    use std::{
        fs::File,
        io,
        os::{fd::AsRawFd, unix::prelude::MetadataExt},
        path::Path,
    };

    use super::{Fs, Os};

    impl DiskBlockSize for Fs {
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
    use std::{
        fs::File,
        io,
        os::windows::{ffi::OsStrExt, fs::OpenOptionsExt},
        path::Path,
    };

    use windows::{
        core::PCWSTR,
        Win32::{Foundation::MAX_PATH, Storage::FileSystem},
    };

    use super::{DiskBlockSize, Fs, Open, OpenOptions};

    impl DiskBlockSize for Fs {
        fn disk_block_size(path: impl AsRef<Path>) -> io::Result<usize> {
            unsafe {
                let mut volume = [0u16; MAX_PATH as usize];

                let mut windows_file_path = path
                    .as_ref()
                    .as_os_str()
                    .encode_wide()
                    .collect::<Vec<u16>>();

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

    impl Open for OpenOptions {
        fn open(mut self, path: impl AsRef<Path>) -> io::Result<File> {
            if self.lock {
                self.inner.share_mode(0);
            }

            if self.bypass_cache {
                let flags =
                    FileSystem::FILE_FLAG_NO_BUFFERING | FileSystem::FILE_FLAG_WRITE_THROUGH;

                self.inner.custom_flags(flags.0);
            }

            self.inner.open(path)
        }
    }
}
