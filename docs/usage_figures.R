# Figures of USAGE.md: 1000 Genomes chr20, K = 5. Run in the directory with the admixer output files:
#   kg.5.Q kg.5.corres.txt kg.5.parental             (all 157,318 SNPs)
#   kgp.5.Q kgp.5.parental kgp.5.paired              (LD-pruned, 28,426 SNPs)
# plus kgall_chr20.fam and the 1000 Genomes panel file (sample, population, superpopulation).
# Usage: Rscript usage_figures.R panel.txt visFuns.R outdir
library(ggplot2)
library(cowplot)
a <- commandArgs(TRUE)
panel <- a[1]; visfuns <- a[2]; out <- a[3]
source(visfuns)  # plotAdmix, plotCorRes (evalAdmix)

pan <- read.table(panel, header = TRUE, comment.char = "")
fam <- read.table("kgall_chr20.fam")
pan <- pan[match(fam$V2, pan$SampleID), ]
stopifnot(!anyNA(pan$Population))
sup_order <- c("AFR", "AMR", "EUR", "SAS", "EAS")
pop <- pan$Population
sup <- factor(pan$Superpopulation, sup_order)
pop_levels <- unique(pop[order(sup, pop)])
popf <- factor(pop, pop_levels)

# Name each ancestry by the superpopulation with the highest mean proportion (AMR: Native American)
anc_names <- c(AFR = "African", AMR = "Native American", EUR = "European", SAS = "South Asian", EAS = "East Asian")
anc_col <- c(African = "#E69F00", "Native American" = "#009E73", European = "#0072B2",
             "South Asian" = "#CC79A7", "East Asian" = "#D55E00")
name_cols <- function(Q) {
  m <- apply(as.matrix(Q), 2, function(q) tapply(q, sup, mean))
  s <- rownames(m)[apply(m, 2, which.max)]
  stopifnot(!anyDuplicated(s))
  anc_names[s]
}

# 1. Admixture proportions (evalAdmix's plotAdmix)
q <- read.table("kg.5.Q")
an <- name_cols(q)
ord <- orderInds(pop = as.character(popf), q = q, popord = pop_levels)
png(file.path(out, "kg_admix.png"), width = 2400, height = 800, res = 200)
par(mar = c(3, 6, 1, 1))
plotAdmix(q, pop = pop, ord = ord, rotatelab = 90, padj = 0.02, cex.lab = 0.8, colorpal = anc_col[an], main = "",
          drawindslines = FALSE)
dev.off()

# 2. evalAdmix correlation of residuals
r <- as.matrix(read.table("kg.5.corres.txt"))
png(file.path(out, "kg_evaladmix.png"), width = 1600, height = 1400, res = 200)
plotCorRes(r, pop = pop, ord = ord, max_z = 0.1, rotatelabpop = 90, adjlab = 0.05, cex.lab = 0.6,
           cex.lab.2 = 0.6, title = "")
dev.off()

# 3. Parental gain per individual, all SNPs vs LD-pruned
gain <- function(f) { p <- read.table(f, header = TRUE); p$loglik_parental - p$loglik_admixture }
g <- rbind(data.frame(set = "all 157,318 SNPs", pop = popf, sup, gain = gain("kg.5.parental")),
           data.frame(set = "LD-pruned, 28,426 SNPs", pop = popf, sup, gain = gain("kgp.5.parental")))
g$set <- factor(g$set, unique(g$set))
lab <- aggregate(gain ~ set + pop, g, function(x) sum(x > 10))
p3 <- ggplot(g, aes(pop, gain + 1, colour = sup)) +
  geom_hline(yintercept = 11, linetype = 2, colour = "grey40") +
  geom_jitter(width = 0.25, height = 0, size = 0.7, alpha = 0.7) +
  geom_text(data = lab, aes(pop, 3e4, label = gain), colour = "black", size = 2.6, inherit.aes = FALSE) +
  facet_wrap(~set, ncol = 1) +
  scale_y_log10(breaks = c(1, 11, 101, 1001, 10001), labels = c(0, 10, 100, 1000, 10000)) +
  scale_colour_manual(values = c(AFR = "#E69F00", AMR = "#009E73", EUR = "#0072B2", SAS = "#CC79A7",
                                 EAS = "#D55E00"), name = NULL) +
  labs(x = NULL, y = "gain: log L parental − log L ADMIXTURE") +
  theme_bw(base_size = 10) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5), legend.position = "top",
        panel.grid.minor = element_blank())
ggsave(file.path(out, "kg_parental_gain.png"), p3, width = 8, height = 5.5, dpi = 200)

# 4. Q and the two parents of the individuals with gain > 10 (pruned data)
qp <- read.table("kgp.5.Q")
anp <- name_cols(qp)
pp <- read.table("kgp.5.parental", header = TRUE)
K <- 5
sel <- which(pp$loglik_parental - pp$loglik_admixture > 10)
sel <- sel[order(popf[sel], -qp[sel, which(anp == "European")])]
long <- function(M, what) data.frame(ind = rep(seq_along(sel), K), pop = rep(popf[sel], K), what = what,
                                     anc = rep(anp, each = length(sel)), p = unlist(M[sel, ]))
d4 <- rbind(long(qp, "Q (ADMIXTURE)"), long(pp[, 1:K], "parent 1"), long(pp[, K + 1:K], "parent 2"))
d4$what <- factor(d4$what, c("Q (ADMIXTURE)", "parent 1", "parent 2"))
d4$anc <- factor(d4$anc, anc_names)
p4 <- ggplot(d4, aes(factor(ind), p, fill = anc)) + geom_col(width = 1) +
  facet_grid(what ~ pop, scales = "free_x", space = "free_x", switch = "y") +
  scale_fill_manual(values = anc_col, name = NULL) + scale_y_continuous(expand = c(0, 0), breaks = NULL) +
  labs(x = NULL, y = NULL) + theme_minimal(base_size = 10) +
  theme(axis.text.x = element_blank(), panel.spacing.x = unit(2, "pt"), legend.position = "top",
        strip.text.y.left = element_text(angle = 0), panel.grid = element_blank())
ggsave(file.path(out, "kg_parental_admixed.png"), p4, width = 8, height = 3.6, dpi = 200)

# 5. Paired ancestry of three individuals: ADMIXTURE's expectation (q_a q_b, twice off the diagonal) and the estimate
pr <- read.table("kgp.5.paired", header = TRUE)
pairs <- do.call(rbind, lapply(1:K, function(a) data.frame(a = a, b = a:K)))
gp <- pp$loglik_parental - pp$loglik_admixture
afreu <- sel[popf[sel] == "ASW" & qp[sel, which(anp == "Native American")] < 0.1]  # African x European
pick <- c(afreu[which.max(gp[afreu])],
          sel[popf[sel] == "PUR"][which.max(gp[sel][popf[sel] == "PUR"])],
          which(popf == "PUR" & gp == 0)[1])
d5 <- do.call(rbind, lapply(pick, function(i) {
  qi <- unlist(qp[i, ])
  e <- ifelse(pairs$a == pairs$b, 1, 2) * qi[pairs$a] * qi[pairs$b]
  who <- sprintf("%s (%s)\ngain: parental %.0f, paired %.0f", pan$SampleID[i], pop[i], gp[i],
                 pr$loglik_paired[i] - pr$loglik_admixture[i])
  rbind(data.frame(who, what = "ADMIXTURE: q_a q_b", pairs, p = e),
        data.frame(who, what = "paired estimate", pairs, p = unlist(pr[i, seq_len(nrow(pairs))])))
}))
ia <- match(anp[d5$a], anc_names); ib <- match(anp[d5$b], anc_names)  # upper triangle in the plotted order
d5$A <- factor(anc_names[pmax(ia, ib)], anc_names); d5$B <- factor(anc_names[pmin(ia, ib)], rev(anc_names))
d5$who <- factor(d5$who, unique(d5$who))
p5 <- ggplot(d5, aes(A, B, fill = p)) + geom_tile(colour = "white") +
  geom_text(aes(label = ifelse(p >= 0.005, sprintf("%.2f", p), "")), size = 2.4) +
  facet_grid(what ~ who) + scale_fill_gradient(low = "white", high = "#0072B2", limits = c(0, 1), name = "probability") +
  labs(x = NULL, y = NULL) + theme_bw(base_size = 9) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank())
ggsave(file.path(out, "kg_paired.png"), p5, width = 9, height = 5, dpi = 200)
cat("individuals in figure 5:", pan$SampleID[pick], "\n")
