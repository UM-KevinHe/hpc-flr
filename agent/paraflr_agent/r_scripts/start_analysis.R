#!/usr/bin/env Rscript
# start_analysis.R — dispatcher for the start_analysis tool (the wizard).
#
# The escape hatch for requests too vague to route: instead of guessing an
# estimator, the agent calls this and asks the user the three questions that
# actually determine the analysis. Returning a fixed checklist from R (rather
# than letting the model improvise one) keeps the vague-request path as
# auditable as the routed one.
#
# input.json : {"user_language": "en"|"zh"}
# output.json: {"status":"ok", "questions": [...], "tools": {...}}

.script_dir <- function() {
  f <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  # Rscript encodes spaces in --file= as "~+~" (paths with spaces hit this).
  if (length(f)) dirname(gsub("~\\+~", " ", sub("^--file=", "", f[1]))) else getwd()
}
source(file.path(.script_dir(), "_common.R"))

QUESTIONS_EN <- list(
  list(key = "data",
       ask = "Where is your data, and which fields hold the binary outcome, the covariates, and the provider identifier? (An .rda, .rds, or .csv path is enough — I will inspect it.)",
       why = "Every paraflr tool needs a 0/1 outcome, a covariate matrix, and one provider id per record."),
  list(key = "goal",
       ask = "Do you want the fitted provider effects for the whole population, or a hypothesis test for one specific provider?",
       why = "Estimation is fit_flr; flagging a provider against a null is test_provider."),
  list(key = "test_choice",
       ask = "If you want a test: which one — Wald, score, or penalised likelihood-ratio — and against what null (the population median, or a fixed value)?",
       why = "The three tests differ most for small or low-event providers, which is where provider profiling usually lives."),
  list(key = "scale",
       ask = "How large is the dataset, and how many cores may I use?",
       why = "logis_firth() parallelises over OpenMP threads; benchmark_threads reports the scaling.")
)

QUESTIONS_ZH <- list(
  list(key = "data",
       ask = "数据在哪里？哪些字段是二分类结局、协变量矩阵和医院/中心编号？（给出 .rda / .rds / .csv 路径即可）",
       why = "所有 paraflr 工具都需要 0/1 结局、协变量矩阵和逐条记录的 provider id。"),
  list(key = "goal",
       ask = "您是想估计所有 provider 的效应，还是对某一个 provider 做假设检验？",
       why = "前者用 fit_flr，后者用 test_provider。"),
  list(key = "test_choice",
       ask = "如果做检验：用 Wald、score 还是惩罚似然比 (LRT)？零假设是总体中位数还是固定值？",
       why = "在小医院、低事件率情形下三种检验差异最大。"),
  list(key = "scale",
       ask = "数据规模大约多少？可以用几个核？",
       why = "logis_firth() 通过 OpenMP 线程并行；benchmark_threads 可以给出加速比。")
)

a <- parse_args()
result <- with_error_payload("start_analysis.R", {
  input <- read_input(a$input)
  lang <- if (is.null(input$user_language)) "en" else tolower(as.character(input$user_language))
  zh <- lang %in% c("zh", "zh-cn", "zh_cn", "chinese", "中文")

  list(status = "ok",
       language = if (zh) "zh" else "en",
       questions = if (zh) QUESTIONS_ZH else QUESTIONS_EN,
       tools = list(
         fit_flr = "Fit Firth-corrected logistic regression with one effect per provider (paraflr::logis_firth).",
         test_provider = "Wald / score / penalised-LRT test of one provider effect against the median or a fixed null (paraflr::test_gamma.single).",
         benchmark_threads = "Time the same fit across OpenMP thread counts and check the estimates agree.",
         simulate_provider_data = "Generate a clustered binary dataset when you have none to hand."),
       note = "Ask these questions, then call the matching tool. Do not fit anything until the data fields are known.")
})
write_result(result, a$output)
