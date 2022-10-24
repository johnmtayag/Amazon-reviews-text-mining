install.packages("tm")
install.packages("SnowballC")
install.packages("syuzhet")
install.packages("ggplot2")

setwd("C:/Users/johnt/Desktop/UExt/Text_Mining")

df_data = read.csv("AllProductReviews.csv", encoding = 'UTF-8')
n_reviews = nrow(df_data)

### Plot number of reviews from each rating

rating = c(1,2,3,4,5)
ratingnum = c()

for (i in rating) {
  ratingnum = c(ratingnum, length(df_data[df_data$ReviewStar == i, 3]) )

}

ratingdata = data.frame(rating = rating, ratingnum = ratingnum)

library(ggplot2)

ggplot(ratingdata, aes(x = '', y = ratingnum, fill = rating, color = rating)) +
  geom_bar(stat = 'identity', color = 'white') +
  coord_polar('y') +
  theme_void()

# Combine the review title and the review body (title often includes overall sentiment)
df_data$Combined = paste(df_data[[1]], df_data[[2]])


### Split data into good and bad reviews
### good review = rating >=4
### bad review = rating < 4

cutoff = 4

### Changing labels for the SVM analysis
df_data$ReviewStar[ df_data$ReviewStar < cutoff] = 0
df_data$ReviewStar[ df_data$ReviewStar >= cutoff] = 1


df_good = df_data[ df_data$ReviewStar == 1, 'Combined']
df_bad = df_data[ df_data$ReviewStar == 0, 'Combined']

rating = c(1,2,3,4,5)
ratingnum = 

library(tm)
library(SnowballC)



build_corpus = function(vec, myStopWords) {
  ### Build a preprocessing function to build a word 
  ### Input: A vector containing all documents, a vector containing stopwords
  ### Output: A word corpus with all preprocessing steps completed
  # Tasks:
  # Create the corpus
  # Strip most/all punctuation
  # Split corpus into individual words and save to own variable
  # Convert to all lowercase
  # Delete all stop words - find an existing list and add anything
  # Stem
  ### Check the stems
  # Set min/max limits
  # Delete numbers (can be ratings, price, product number, date; probably not useful)
  
  ### Create the corpus
  vec_corp = VCorpus(VectorSource(vec))
  
  ### Remove punctuations
  # Replace ',' '.' '/' with spaces first to avoid accidental word merges (ex: good.You --> goodYou)
  vec_corp = tm_map(vec_corp, content_transformer(gsub), pattern = ',', replacement = ' ')
  vec_corp = tm_map(vec_corp, content_transformer(gsub), pattern = '\\.', replacement = ' ')
  vec_corp = tm_map(vec_corp, content_transformer(gsub), pattern = '\\/', replacement = ' ')
  
  # Remove UTF-8 errors
  #vec_corp = tm_map(vec_corp, content_transformer(gsub), pattern = '\\<U.+?\\>', replacement = ' ')
  
  vec_corp = tm_map(vec_corp, removePunctuation)
  
  ### Remove numbers
  vec_corp = tm_map(vec_corp, removeNumbers)
  
  ### Convert to lower case
  vec_corp = tm_map(vec_corp, content_transformer(tolower))
  
  ### Remove stopwords
  vec_corp = tm_map(vec_corp, removeWords, tolower(myStopWords))
  
  ### Perform stemming
  vec_corp = tm_map(vec_corp, stemDocument)
  
  ### Eliminate extra whitespaces
  vec_corp = tm_map(vec_corp, stripWhitespace)
  
  return(vec_corp)
}

# myStopWords allows adding custom stopwords to the existing list
products = tolower(unique(df_data$Product))
custom = c('u', 'ur', 'headphone', 'headphones', 'earphone', 'earphones')
myStopWords = c(stopwords('en'), custom, products)


# all includes all reviews; good/bad splits them into respective categories
#review_corp_all = build_corpus(df_data$Combined, myStopWords)
review_corp_good = build_corpus(df_good, myStopWords)
review_corp_bad = build_corpus(df_bad, myStopWords)

#inspect(review_corp_all[[5]])


### Build the DTM

# ctrl_tf and ctrl_tfidf are controls that include upper and lower term frequency bounds
ctrl_tf = list(weighting = weightTf, bounds = list(local = c(0,Inf)))
ctrl_tfidf = list(weighting = weightTfIdf, bounds = list(local = c(0,Inf)))

#R: review, D/T: DTM/TDM, G/B: good/bad, tf/tfidf: termfreq/termfreq-inversedocumentfreq
#rDGtf_bounded = DocumentTermMatrix(review_corp_good, control = ctrl_tf)
rDGtf = DocumentTermMatrix(review_corp_good, control = list(weighting = weightTf))
rDBtf = DocumentTermMatrix(review_corp_bad, control = list(weighting = weightTf))
#Test to compare outputs for TF-IDF
rDGtfidf = DocumentTermMatrix(review_corp_good, control = list(weighting = weightTfIdf))
rDBtfidf = DocumentTermMatrix(review_corp_bad, control = list(weighting = weightTfIdf))


## Remove terms above some sparse level
sparse_threshold = 0.98 #This means words must appear in at least 1/10 of the reviews to be included in the TDM/DTM
rDGtf2 = removeSparseTerms(rDGtf, sparse_threshold)
rDBtf2 = removeSparseTerms(rDBtf, sparse_threshold)
rDGtfidf2 = removeSparseTerms(rDGtfidf, sparse_threshold)
rDBtfidf2 = removeSparseTerms(rDBtfidf, sparse_threshold)
#inspect(rDGtf2)

#Get dataframe versions

#Convert to matrix to sum the occurences per document
num = 20

#### GOOD, TF
#rDGtf2_mat = as.matrix(rDGtf2)
rDGtf2_df = data.frame(as.matrix(rDGtf2))
good_freqtf = sort(colSums(rDGtf2_df), decreasing = TRUE)
good_freqheadtf = head(good_freqtf, num)
#goodtf_df = data.frame(terms = names(good_freqtf), count = good_freqtf)
goodtfhead_df = data.frame(terms = names(good_freqheadtf), count = good_freqheadtf)

### GOOD, TFIDF
#rDGtfidf2_mat = as.matrix(rDGtfidf2)
rDGtfidf2_df = data.frame(as.matrix(rDGtfidf2))
good_freqtfidf = sort(colSums(rDGtfidf2_df), decreasing = TRUE)
good_freqheadtfidf = sort(head(good_freqtfidf, num), decreasing = TRUE)
goodtfidf_df = data.frame(terms = names(good_freqtfidf), count = good_freqtfidf)
goodtfidfhead_df = data.frame(terms = names(good_freqheadtfidf), count = good_freqheadtfidf)

### BAD, TF
#rDBtf2_mat = as.matrix(rDBtf2)
rDBtf2_df = data.frame(as.matrix(rDBtf2))
bad_freqtf = sort(colSums(rDBtf2_df), decreasing = TRUE)
bad_freqheadtf = sort(head(bad_freqtf, num), decreasing = TRUE)
#badtf_df = data.frame(terms = names(bad_freqtf), count = bad_freqtf)
badtfhead_df = data.frame(terms = names(bad_freqheadtf), count = bad_freqheadtf)

### BAD, TFIDF
#rDBtfidf2_mat = as.matrix(rDBtfidf2)
rDBtfidf2_df = data.frame(as.matrix(rDBtfidf2))
bad_freqtfidf = sort(colSums(rDBtfidf2_df), decreasing = TRUE)
bad_freqheadtfidf = sort(head(bad_freqtfidf, num), decreasing = TRUE)
badtfidf_df = data.frame(terms = names(bad_freqtfidf), count = bad_freqtfidf)
badtfidfhead_df = data.frame(terms = names(bad_freqheadtfidf), count = bad_freqheadtfidf)


## Plotting 15 most common words from the reviews per category
library(ggplot2)

ggplot(data = goodtfhead_df, aes(terms, count)) +
  geom_col(color = 'deepskyblue', fill = 'deepskyblue' , width = .8) +
  labs(title = '15 most weighted words in good reviews (TF)') +
  theme(axis.text.x = element_text(angle = 45))

ggplot(data = goodtfidfhead_df, aes(terms, count)) +
  geom_col(color = 'deepskyblue', fill = 'deepskyblue' , width = .8) +
  labs(title = '15 most weighted words in good reviews (TF-IDF)') +
  theme(axis.text.x = element_text(angle = 45))

ggplot(data = badtfhead_df, aes(terms, count)) +
  geom_col(color = 'darkorange', fill = 'darkorange' , width = .8) +
  labs(title = '15 most weighted words in bad reviews (TF)') +
  theme(axis.text.x = element_text(angle = 45))

ggplot(data = badtfidfhead_df, aes(terms, count)) +
  geom_col(color = 'darkorange', fill = 'darkorange' , width = .8) +
  labs(title = '15 most weighted words in bad reviews (TF-IDF)') +
  theme(axis.text.x = element_text(angle = 45))

#Comparing most weighted terms from good/bad in a dataframe



bothhead_df = data.frame(g_terms = goodtfidfhead_df$terms, 
                         g_count = goodtfidfhead_df$count,
                         b_terms = badtfidfhead_df$terms, 
                         b_count = badtfidfhead_df$count)

gwordlist = goodtfidf_df$terms
bwordlist = badtfidf_df$terms

words_in_both = intersect(gwordlist, bwordlist)
only_good = setdiff(gwordlist, bwordlist)
only_bad = setdiff(bwordlist, gwordlist)

cat('Number of words in good: ', nrow(goodtfidf_df), 
  '\nNumber of words in bad: ', nrow(badtfidf_df),
  '\nNumber of words in both sets: ', length(words_in_both),
  '\n')
print(words_in_both)

cat('Number of words only in good: ', length(only_good),
    '\nNumber of words only in bad: ', length(only_bad))

badtfidf_df$inboth = 1
goodtfidf_df$inboth = 1

badtfidf_df[badtfidf_df$terms %in% only_bad, 'inboth'] = 7
goodtfidf_df[goodtfidf_df$terms %in% only_good, 'inboth'] = 7

badtfidf_df$inboth_count = badtfidf_df$count * badtfidf_df$inboth + 1
goodtfidf_df$inboth_count = goodtfidf_df$count * goodtfidf_df$inboth + 1

### Making word clouds with the unique words in either set
install.packages('wordcloud')
install.packages('RColorBrewer')

library(wordcloud)
library(RColorBrewer)

set.seed(100)

#All terms from good set; unique terms in red
wordcloud(words = goodtfidf_df$terms, 
          freq = goodtfidf_df$count, 
          random.order = FALSE,
          rot.per = .35,
          ordered.colors = TRUE,
          colors = brewer.pal(8, 'Set1')[factor(goodtfidf_df$inboth)])

#Highlight unique terms from the good set
wordcloud(words = goodtfidf_df$terms, 
          freq = goodtfidf_df$inboth_count,
          #scale = c(2.5, .5),
          random.order = FALSE,
          rot.per = .3,
          ordered.colors = TRUE,
          colors = brewer.pal(8, 'Set1')[factor(goodtfidf_df$inboth)])

#All terms from bad set; unique terms in red
wordcloud(words = badtfidf_df$terms, 
          freq = badtfidf_df$count, 
          random.order = FALSE,
          rot.per = .35,
          ordered.colors = TRUE,
          colors = brewer.pal(8, 'Set1')[factor(badtfidf_df$inboth)])

#Highlight unique terms from the bad set
wordcloud(words = badtfidf_df$terms, 
          freq = badtfidf_df$inboth_count,
          #scale = c(2.5, .5),
          random.order = FALSE,
          rot.per = .3,
          ordered.colors = TRUE,
          colors = brewer.pal(8, 'Set1')[factor(badtfidf_df$inboth)])


### SVM: Predict good vs bad review

install.packages('e1071')
library(e1071)

# TESTING - SPLIT DATA INTO QUARTERS



## Split data into training and test sets
sample_size = floor(0.7 * nrow(df_data))
set.seed(100)

df_train = df_data[sample(nrow(df_data), sample_size),]
df_test = df_data[-sample(nrow(df_data), sample_size),]

# df_test and df_train contain the data
# $Combined: words -> create a corpus and DTM, convert to data frame
# $ReviewStar: 1 (good) or 0 (bad)

# myStopWords allows adding custom stopwords to the existing list
#products = tolower(unique(df_data$Product))
#custom = c('u', 'ur', 'headphone', 'headphones', 'earphone', 'earphones')
#myStopWords = c(stopwords('en'), custom, products)

train_corp = build_corpus(df_train$Combined, myStopWords)
test_corp = build_corpus(df_test$Combined, myStopWords)

### Build the DTM

# TF-IDF control
#ctrl_tfidf = list(weighting = weightTfIdf, bounds = list(local = c(0,Inf)))

## Remove terms above some sparse level for training set
sparse_threshold = 0.98 #This means words must appear in at least 1/10 of the reviews to be included in the TDM/DTM
test_dtm = removeSparseTerms(rDGtf, sparse_threshold)

# TRAIN DTM to DF
train_dtm = DocumentTermMatrix(train_corp, control = list(weighting = weightTfIdf))
train_dtm2 = removeSparseTerms(train_dtm, sparse_threshold)
train_dtm_df = data.frame(as.matrix(train_dtm2))
# TEST DTM to DF
test_dtm = DocumentTermMatrix(test_corp, control = list(weighting = weightTfIdf))
test_dtm_df = data.frame(as.matrix(test_dtm))

# Compile dataframe variables
train_dat = data.frame(train_dtm_df, ReviewStar = as.factor(df_train$ReviewStar))
test_dat = data.frame(test_dtm_df, ReviewStar = as.factor(df_test$ReviewStar))

### Run SVM

###FINDING THE BEST PARAMETERS
#test_svm = tune.svm(ReviewStar~., data = test_dat, kernel = 'radial',
#                    cost = .1:10, gamma = .01:1)

# From test_svm, the best parameters are gamma = 0.01 and cost = 9.1
g = 0.01 #gamma
c = 9.1 #cost

# Train the model
my_model_rad = svm(ReviewStar~., data = train_dat, kernel = 'radial', cost = c, gamma = g)
my_model_sig = svm(ReviewStar~., data = train_dat, kernel = 'sigmoid', cost = c)
my_model_lin = svm(ReviewStar~., data = train_dat, kernel = 'linear', cost = c)

# Predict
pred_rad = predict(my_model_rad, test_dat)
pred_sig = predict(my_model_sig, test_dat)
pred_lin = predict(my_model_lin, test_dat)

confmat_rad = table(pred_rad, test_dat$ReviewStar)
confmat_sig = table(pred_sig, test_dat$ReviewStar)
confmat_lin = table(pred_lin, test_dat$ReviewStar)

print(confmat_rad)
print(confmat_sig)
print(confmat_lin)

ml_stats_2x2 = function(mat) {
  # Works for a 2x2 confusion matrix - positive = 1, negative = 0
  class1 = mat[,1]
  class2 = rev(mat[,2])
  
  #c[1] = # from the class classified correctly
  #c[2] = # from the class classified incorrectly
  
  sum1 = sum(class1)
  sum2 = sum(class2)
  
  accuracy1 = class1[1]/sum1
  accuracy2 = class2[1]/sum2
  
  precision = class1[1] / (class1[1] + class2[1])
  recall = class1[1] / (sum1)
  fscore = 2 * ((precision * recall) / (precision + recall))
  
  output = data.frame(sum1 = sum1, sum2 = sum2,
                      accuracy1 = accuracy1, accuracy2 = accuracy2,
                      TP = class1[1]/sum1, FP = class2[2]/sum2,
                      fscore = fscore,
                      precision = precision, recall = recall)
  
  return(output)
  
}

ml_stats_2x2(confmat_rad)
ml_stats_2x2(confmat_sig)
ml_stats_2x2(confmat_lin)



#findAssocs(rDBtf2, 'sound', .7)




