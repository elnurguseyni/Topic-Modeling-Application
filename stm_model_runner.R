# run_stm.R

# Force Rscript to use the user's site-library path
.libPaths(c("/opt/homebrew/lib/R/4.5/site-library", .libPaths()))

# Auto-install required packages if not present
required_packages <- c("stm", "readr", "dplyr")
installed_packages <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(stm)
library(readr)
library(dplyr)

# Load input from Streamlit
input <- read_csv("stm_input_docs.csv")

# Preprocess for STM
processed <- textProcessor(documents = input$doc, metadata = input)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)

# Load number of topics from config
k <- as.integer(readLines("stm_config.txt"))


# Early validation for documents and vocabulary size
if (length(out$documents) < 10) {
  stop("Too few valid documents remain after preprocessing. Try a different dataset or relax preprocessing.")
}
if (length(out$vocab) < 10) {
  stop("Too few unique words after preprocessing. Vocabulary is too small to model topics.")
}

# Fit STM model using metadata
model <- stm(documents = out$documents, vocab = out$vocab, 
             K = k, prevalence = ~ meta,
             data = out$meta, init.type = "LDA", max.em.its = 75)

# Get top words per topic
topic_words <- labelTopics(model, n = 10)
topic_df <- data.frame(
  Topic = 1:nrow(topic_words$prob),
  TopWords = apply(topic_words$prob, 1, function(row) paste(na.omit(row), collapse = ", "))
)
write_csv(topic_df, "stm_topics_output.csv")

# Estimate metadata effect
effect <- estimateEffect(1:nrow(topic_words$prob) ~ meta, model, metadata = out$meta, uncertainty = "Global")
effect_summary <- data.frame(summary(effect)$tables[[1]])
write_csv(effect_summary, "stm_metadata_effect.csv")
