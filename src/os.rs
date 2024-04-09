//! OS specific code.

use std::{
    fs::{self, File},
    io,
    path::Path,
};

/// Necessary syscalls to get the file system block size.
pub(crate) trait FileSystemBlockSize {
    /// Returns the underlying IO device block size or preferred IO buffer size.
    fn block_size(path: impl AsRef<Path>) -> io::Result<usize>;
}

/// Trait for opening a file depending on the OS.
pub(crate) trait Open {
    /// Opens the file with the specified [`OpenOptions`].
    fn open(self, path: impl AsRef<Path>) -> io::Result<File>;
}

/// Used to implement [`FileSystemBlockSize`] and [`Open`] traits on different
/// operating systems.
pub(crate) struct Fs;

impl Fs {
    /// See [`OpenOptions`].
    pub fn options() -> OpenOptions {
        OpenOptions::default()
    }
}

/// Just like [`fs::OpenOptions`] but with some additional options that are
/// handled differently depending on the operating system.
///
/// See the resources below for some background.
///
/// Linux:
/// - [What does O_DIRECT really mean?](https://stackoverflow.com/questions/41257656/what-does-o-direct-really-mean)
/// - [O_SYNC VS O_DSYNC](https://www.linuxquestions.org/questions/linux-general-1/o_sync-vs-o_dsync-4175616852/)
/// - [How are the O_SYNC and O_DIRECT flags in open(2) different/alike?](https://stackoverflow.com/questions/5055859/how-are-the-o-sync-and-o-direct-flags-in-open2-different-alike)
/// - [open(2) man page](https://man7.org/linux/man-pages/man2/open.2.html)
///
/// Windows:
/// - [CreateFileA function parameters](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea#parameters)
/// - [CreateFile with FILE_FLAG_NO_BUFFERING but not FILE_FLAG_WRITE_THROUGH](https://stackoverflow.com/questions/51315745/createfile-with-file-flag-no-buffering-but-not-file-flag-write-through)
/// - [On the interaction between the FILE_FLAG_NO_BUFFERING and FILE_FLAG_WRITE_THROUGH flags](https://devblogs.microsoft.com/oldnewthing/20210729-00/?p=105494)
/// - [How Postgres uses Windows flags](https://github.com/postgres/postgres/blob/a767cdc84c9a4cba1f92854de55fb8b5f2de4598/src/port/open.c#L84-L100)
pub(crate) struct OpenOptions {
    /// Inner [`std::fs::OpenOptions`] instance.
    inner: fs::OpenOptions,
    /// Bypasses OS cache.
    bypass_cache: bool,
    /// When calling write() makes sure that data reaches disk before returning.
    sync_on_write: bool,
    /// Locks the file exclusively for the calling process.
    lock: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            inner: File::options(),
            bypass_cache: false,
            sync_on_write: false,
            lock: false,
        }
    }
}

impl OpenOptions {
    /// Disables the OS cache for reads and writes.
    pub fn bypass_cache(mut self, bypass_cache: bool) -> Self {
        self.bypass_cache = bypass_cache;
        self
    }

    /// Every call to write() will make sure that the data reaches the disk.
    pub fn sync_on_write(mut self, sync_on_write: bool) -> Self {
        self.sync_on_write = sync_on_write;
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
        os::{
            fd::AsRawFd,
            unix::{fs::OpenOptionsExt, prelude::MetadataExt},
        },
        path::Path,
    };

    use super::{FileSystemBlockSize, Fs, Open, OpenOptions};

    impl FileSystemBlockSize for Fs {
        fn block_size(path: impl AsRef<Path>) -> io::Result<usize> {
            Ok(File::open(&path)?.metadata()?.blksize() as usize)
        }
    }

    impl Open for OpenOptions {
        fn open(mut self, path: impl AsRef<Path>) -> io::Result<File> {
            let mut flags = 0;

            if self.bypass_cache {
                flags |= libc::O_DIRECT;
            }

            if self.sync_on_write {
                flags |= libc::O_DSYNC;
            }

            if flags != 0 {
                self.inner.custom_flags(flags);
            }

            let file = self.inner.open(&path)?;

            if self.lock {
                let lock = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };

                if lock != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("could not lock file {}", path.as_ref().display()),
                    ));
                }
            }

            Ok(file)
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

    impl FileSystemBlockSize for Fs {
        fn block_size(path: impl AsRef<Path>) -> io::Result<usize> {
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
            let mut flags = FileSystem::FILE_FLAGS_AND_ATTRIBUTES(0);

            if self.bypass_cache {
                flags |= FileSystem::FILE_FLAG_NO_BUFFERING;
            }

            if self.sync_on_write {
                flags |= FileSystem::FILE_FLAG_WRITE_THROUGH;
            }

            if flags.0 != 0 {
                self.inner.custom_flags(flags.0);
            }

            if self.lock {
                self.inner.share_mode(0);
            }

            self.inner.open(path)
        }
    }
}
