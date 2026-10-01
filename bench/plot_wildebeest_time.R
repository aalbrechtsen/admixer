# README figure docs/wildebeest_time_admixture_vs_admixer.png: wall time per run on blue_wildebeest_noLD, K = 7,
# seeds 1-10, 8 threads, one run at a time.
# usage: Rscript --no-init-file bench/plot_wildebeest_time.R ADMIXTURE_DIR ADMIXER_RESULTS OUT.png
#   ADMIXTURE_DIR: s1..s10/t.txt with the wall time of each ADMIXTURE 1.3.0 run (data/conv_test/admixture)
#   ADMIXER_RESULTS: lines "seed loglik wall cpu iters" (admix_prog/bench2/result_wb_admixer.txt)
suppressMessages(library(ggplot2))
a <- commandArgs(TRUE)
adm <- sapply(1:10, function(s) as.numeric(readLines(file.path(a[1], paste0("s", s), "t.txt"))))
ax <- read.table(a[2], comment.char = "#", col.names = c("seed", "loglik", "wall", "cpu", "iters"))
ax <- ax[ax$seed %in% 1:10, ]
d <- data.frame(program = factor(rep(c("ADMIXTURE 1.3.0", "admixer"), each = 10),
                                 levels = c("ADMIXTURE 1.3.0", "admixer")),
                wall = c(adm, ax$wall))
cat(sprintf("median wall: ADMIXTURE %.1f s, admixer %.2f s, ratio %.0f\n", median(adm), median(ax$wall),
            median(adm) / median(ax$wall)))
set.seed(1)
p <- ggplot(d, aes(program, wall)) +
  geom_boxplot(outlier.shape = NA, width = 0.5) +
  geom_jitter(width = 0.1, height = 0, size = 2, colour = "grey30") +
  scale_y_log10(breaks = c(1, 2, 5, 10, 20, 50, 100)) +
  labs(title = "Time per run (8 threads)", x = NULL, y = "wall time per run (s, log scale)") +
  theme_bw(base_size = 11)
ggsave(a[3], p, width = 4, height = 4, dpi = 130)
