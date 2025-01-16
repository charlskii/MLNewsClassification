# Install and Load Required Libraries -------------------------------------
install.packages("tm")
install.packages("SnowballC")
install.packages("dplyr")
install.packages("readr")  
install.packages("tidytext")
install.packages("glmnet")
install.packages("Matrix")
install.packages("caret")
install.packages("randomForest")
install.packages("e1071")
library(tm)
library(SnowballC)
library(dplyr)
library(readr)
library(tidytext)
library(glmnet)
library(Matrix)
library(caret)
library(randomForest)
library(ggplot2)
library(tidyr)
library(e1071)
##Data Preparation

NEWS21 <- read_csv("NEWS.csv")

##Select specific columns
news_cleaning1 <- NEWS21[, c('date', 'source', 'title', 'content', 'category_level_1', 'category_level_2')]

##Assign sources into Left, Right, or Neutral
news_cleaning <- news_cleaning1 %>% 
  mutate(source_type = case_when(
    source %in% c("vox", "theguardianuk", "thenewyorktimes", "npr", "politico") ~ "Liberal",
    source %in% c("washingtonexaminer", "foxnews", "dailycaller", "breitbart") ~ "Conservative",
    source %in% c("wallstreetjournal", "nbcnews", "univision", "usatoday", "abcnews", "cbsnews", "newyorkpost") ~ "Moderate",
    TRUE ~ NA_character_
  )) %>%
  mutate(source_type = as.factor(source_type)) %>%
  na.omit()

##Pie chart
news_cleaning %>%
  count(source_type) %>%  
  ggplot(aes(x = "", y = n, fill = source_type)) +  
  geom_bar(stat = "identity", width = 1) +  
  coord_polar(theta = "y") +  
  labs(title = "Political Standpoint of Articles by News Source",
       fill = "Source Type",
       y = NULL, x = NULL) +
  theme_void() 

##Preprocess the title column
corpus <- Corpus(VectorSource(news_cleaning$title))
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stemDocument)

##Create a dtm
dtm_unigram <- DocumentTermMatrix(corpus)

##Convert the dtm to a data frame
df_unigram <- as.data.frame(as.matrix(dtm_unigram))
df_unigram$source_type <- news_cleaning$source_type

df_tidy <- df_unigram %>%
  pivot_longer(-source_type, names_to = "word", values_to = "count") %>%
  filter(count > 0, word != "", word != "–")


##Find the 20 most frequently used words by source type
top_words <- df_tidy %>%
  group_by(source_type, word) %>%
  summarise(total_count = sum(count), .groups = "drop") %>%
  arrange(source_type, desc(total_count)) %>%
  group_by(source_type) %>%
  slice_head(n = 20)

##Create a bar plot
ggplot(top_words, aes(x = reorder(word, total_count), y = total_count, fill = source_type)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~source_type, scales = "free_y") +  # Separate facets for each source type
  coord_flip() +
  labs(
    title = "Top 20 Most Frequent Words by Source Type",
    x = "Words",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

##Split data into training and testing 
set.seed(1234)
train_index <- createDataPartition(df_unigram$source_type, p = 0.7, list = FALSE)
train_data <- df_unigram[train_index, ]
test_data <- df_unigram[-train_index, ]
train_x <- train_data %>% select(-source_type)
train_y <- train_data$source_type
test_x <- test_data %>% select(-source_type)
test_y <- test_data$source_type

##Logistic regression with lasso reg
lasso_model <- glmnet(
  x = as.matrix(train_x),
  y = train_y,
  family = "multinomial",
  alpha = 1
)

cv_lasso <- cv.glmnet(
  x = as.matrix(train_x),
  y = train_y,
  family = "multinomial",
  alpha = 1
)

best_lambda <- cv_lasso$lambda.min
lasso_predictions <- predict(lasso_model, newx = as.matrix(test_x), s = best_lambda, type = "class")
lasso_conf_matrix <- confusionMatrix(factor(lasso_predictions, levels = levels(test_y)), test_y)
lasso_conf_df <- as.data.frame.table(lasso_conf_matrix$table)
lasso_conf_df$Test <- "Logistic Regression"

##SVM Model
svm_model <- svm(
  x = as.matrix(train_x),
  y = train_y,
  kernel = "linear",
  cost = 1,
  scale = TRUE
)

svm_predictions <- predict(svm_model, as.matrix(test_x))
svm_conf_matrix <- confusionMatrix(factor(svm_predictions, levels = levels(test_y)), test_y)
svm_conf_df <- as.data.frame.table(svm_conf_matrix$table)
svm_conf_df$Test <- "Support Vector Machine"

##RFM

rf_model <- randomForest(
  x = train_x,
  y = train_y,
  ntree = 100,
  mtry = sqrt(ncol(train_x)),
  importance = TRUE
)

rf_predictions <- predict(rf_model, newdata = test_x)
rf_conf_matrix <- confusionMatrix(factor(rf_predictions, levels = levels(test_y)), test_y)
rf_conf_df <- as.data.frame.table(rf_conf_matrix$table)
rf_conf_df$Test <- "Random Forest"

##Combine confusion matrices

combined_conf_df <- bind_rows(lasso_conf_df, svm_conf_df, rf_conf_df)

##Visualise all confusion matrices together

ggplot(combined_conf_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "blue") +
  facet_wrap(~Test) +
  labs(
    title = "Confusion Matrices for Different Models",
    x = "True Labels",
    y = "Predicted Labels",
    fill = "Frequency"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
  theme_minimal()

