# Hang after "Writing output files." (2026-10-09)

## Summary

admixer runs hung for hours after printing `Writing output files.`. The cause
is **not** admixer's code. 128 KiB of the binary file `admixer` (in this
folder, on the NFS mount `/kellyData`) cannot be read: every read of those
pages blocks forever in the kernel. The hang shows up in the evalAdmix step
because that is the first time the program runs code from those pages. The
same binary from `dist/`, unpacked on local disk, runs normally and gives
identical results.

## What was seen

- 7 runs in parallel (`admixer -j 8 --conv 3 rd_ld_admix.bed K`, K = 2..8;
  96 reindeer, 158,115 SNPs; output on the NFS mount `/steveData`):
  - each fit took 3–30 s;
  - `.Q`, `.P.gz`, `.conv` and `.log` were all written by 10:20;
  - the log ends with `Writing output files.`;
  - `.corres.txt` was never written.
- 2 h 39 min later all 7 processes were in state `D`, waiting in the kernel
  (`wchan` = `folio_wait_bit_common`). They had used 1–2 min of CPU and their
  RSS was 76–146 MB.
  - `timeout` (SIGTERM) does not end such a process; `kill -9` does.
- Outputs on local disk (`/tmp`) do not help: the same command hangs in the
  same place.
- With `--no-evaladmix` every run completes (K = 2..8, 91 animals, all under
  30 s), and so does projection (`-P`).

## Diagnosis

1. During the hang the main thread is in `D` / `folio_wait_bit_common` and
   the 7 OpenMP threads are idle (`futex_do_wait`). The process holds only its
   log and stdout open, so it is not blocked on a data or output file.
2. `cp admixer /tmp/...` hangs in the same state, and the partial copy is
   exactly 10,485,760 bytes (10 MiB).
3. Reading the file in blocks (`dd`, under `timeout -s KILL`) narrows the
   problem down:
   - 1 MiB blocks at 9, 11, 12, 15 and 19 MiB read fine; the block at 10 MiB
     hangs;
   - in 64 KiB steps from 10 MiB, only offsets **10 MiB + 512 KiB to
     10 MiB + 640 KiB** hang (file offsets 0xA80000–0xAA0000).
4. That range is inside `.text`:

   ```
   [13] .text  addr 0x408000  offset 0x8000  size 0x12b0390
   ```

   It is program code (virtual address about 0xE80000–0xEA0000). The binary
   is stripped, so the function cannot be named, but the code is first used
   after the outputs are written, i.e. by evalAdmix. The binary is static, so
   that code could also be BLAS or OpenMP code that only evalAdmix calls.
5. The file itself is unchanged: 20,622,248 bytes, last modified
   2026-10-01 16:25. No build was running at the time.
6. Elsewhere the NFS mounts answered normally (`stat` about 5 ms on both).
   Other processes on the machine, two `chrome` processes, were stuck in the
   same `folio_wait_bit_common` state at the same time. That points to the
   NFS client page cache of this machine (kernel 6.17.0-22-generic, NFS 4.2,
   `hard` mounts), not to the file's data on the server.
7. Check: `dist/admixer-linux-x86_64.tar.gz` (also 0.2.1), unpacked to
   `/tmp` and run on the same data with evalAdmix on:
   - finishes in 3.1 s and writes `.corres.txt`;
   - gives `.Q` identical to that of the hung run (max |dQ| = 0).

The likely chain:
- the first run's page-in of these pages over NFS never completed;
- the page-cache entries (folios) stay locked;
- every later reader of the same pages waits on that lock: the 7 parallel
  runs, `cp` and `dd`.

This needs root (or a reboot) to confirm and clear. Normal users cannot see
the kernel log (`dmesg`, `journalctl -k`) or the stack in `/proc/PID/stack`,
and ptrace is blocked (yama), so no backtrace was taken.

## Workarounds

- Use a fresh copy of the binary in a new file (a new inode, so the stuck
  page-cache entries are not used), e.g. unpack `dist/` to local disk, or
  rebuild after moving the old file aside:

  ```
  mv admixer admixer.stuck
  make            # or unpack dist/admixer-linux-x86_64.tar.gz
  ```

  Do not `cp admixer` (it hangs) and do not delete `admixer.stuck` while
  processes may still hold it.
- Until then, `--no-evaladmix` avoids the stuck code for runs that do not need
  `.corres.txt`.
- For the system administrator:
  - stuck folios on `kelly:/kellyData` (NFS 4.2, hard mount) on this client;
  - processes in `D` with `folio_wait_bit_common` (admixer, cp, dd, chrome);
  - `echo 3 > /proc/sys/vm/drop_caches` will not free locked folios; a remount
    or reboot may be needed.

## Fix applied (2026-10-12)

1. **Old binary moved aside.** It is now `admixer.stuck`; do not delete it
   while a process may still hold it.
2. **First rebuild hung.** `make release` was run in this folder (on
   `/kellyData`). `cc1plus` sat for about 70 h with 0.2 s of CPU (the same
   disk stall, while reading or writing on the NFS mount) until it was
   killed. No binary was produced.
3. **Second rebuild worked, on local disk.**
   - Built from `src/`, `Makefile` and `.build/deps` copied to local disk
     (`make release`, with the cached static OpenBLAS 0.3.28 and zlib; GCC 13.3),
     then stripped. It took 15 s.
   - The new binary is 20,693,904 bytes, needs only glibc at run time, and is
     built from HEAD `b9cd489` (2026-10-02). The old one was built on
     2026-10-01, so it is slightly newer.
4. **Installed and checked.**
   - Copied back to `admixer`; `cmp` is identical to the local build.
   - With evalAdmix on, K = 2 and K = 3, run both from local disk and from
     `/kellyData`, finish normally and write `.corres.txt`.
   - Q at K = 2 is identical to that of the `dist` binary (max |dQ| = 0).

If other files on `/kellyData` show the same stall (a process in `D` with
`folio_wait_bit_common`), the same trick works: copy or build on local
disk, or write a new file in place of the stuck one. The mount itself still
needs the administrator.

## Possible hardening in admixer (optional)

admixer has no bug here, but two cheap changes would make such failures
clearer:
- print `evalAdmix ...` before starting evalAdmix, so that a hang is placed
  in the right step;
- flush the log after each line (the log already ended in the right place
  here).

## Files

- Hung run: `/steveData/anders/relatePCA_reindeer/new/admix/` (`admixer*.out`,
  `rd_ld_admix.K.*`).
- Reproduction and the working `dist` run: the session scratch folder
  (`adm/loc2.out` hung, `adm/db2.out` + `adm/db.2.corres.txt` worked).
