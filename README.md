# Analyzing Customer Reviews from Amazon

## Analysis of the data and Rating prediction

The purpose of this project is to analyze customer reviews of different headphones listings on Amazon. The reviews are split into "good" and "bad" sets, 
and the review texts are processed into tokens consisting of individual words. These form the "good" and "bad" review corpuses that are to be analyzed 
to determine the most important words in either good and bad review sets. They are also used to predict if a review will be "good" or "bad."

Binning the data into good and bad categories simplifies the analysis and also removes the skew toward 5 star ratings. Also, from a business's perspective, 
customer satisfaction is key - any rating lower than perfect can be considered bad. If the customer left a low rating, they were unsatisfied with something, 
and discovering these issues is important for guiding product development.

> *Good Reviews*: 4-5 star ratings, represented in the data as 1<br>
> *Bad Reviews*: 1-3 star ratings, represented in the data as 0

This analysis has many real-world applications. While customers can manually include ratings for the product in their reviews, this rating is only an overall statement

## Dataset

The dataset used to create the model is called “Amazon Earphones Reviews” and was posted by user Shital Kat on Kaggle. It is a 2.5MB csv file that consists of 
14,337 reviews and 4 columns:

>Title of review<br>
>Body of review<br>
>Rating (1-5)<br>
>Product name

Link: https://www.kaggle.com/shitalkat/amazonearphonesreviews

## Packages used

>*tm*: Text mining package to build the document-term matrices (DTM)<br>
>*SnowballC*: Contains the base stopwords dictionary<br>
>*ggplot2*: For data visualization tools<br>
>*wordcloud*: For building wordclouds<br>
>*RColorBrewer*: For additional color palettes for plots<br>
>*e1071*: For building SVM prediction models

## Conclusions for the overall review DTM

Without most words referring to preference/sentiment (removed in the stopwords step), a few words regarding headphone quality appear to be the most 
discussed across all reviews. Some conclusions can be made, but for a more accurate analysis, the context of these words would need to be considered:

Most important word:<br>
*product*: Most likely referring to the product being reviewed, but could indicate other products being compared

Other important word categories:<br>
>Various terms indicate focus on build quality:
>Various terms indicate focus on sound quality:
>Various terms indicate focus on product features:
>Various terms indicate focus on customer experience:
>Various terms indicate focus on price and product value:

## Conclusions from the initial analysis of good/bad reviews

The most notable finding is that "good" is ranked as the most important term in both good and bad reviews. This is unexpected, considering that customers who 
leave bad reviews wouldn't normally conclude that the item was "good." There are two likely reasons for this discrepancy

The bad reviews include 3-star reviews which fall in the middle of the spectrum - some aspects of the product may be good, but overall, the customer was disappointed
These reviews may be about how a product is actually "not good" instead of "good"

To account for this, I adjusted the weighting system to emphasize tokens present only in either the good or bad dataset and penalize tokens shared between the two 
(with the penalty relative to how frequent the token is present in the other dataset). This improved the results greatly and I created visualizations to reflect 
these significant changes. 

However, due to time constraints, I was not able to implement this in the following machine learning model and instead used to default TFIDF weighting method.

## Maching Learning predictions

I built a simple SVM model to test if the review text dataset was enough to predict if a given review text was good or bad. A use case for this would be sorting 
unlabeled reviews into the respective categories. It would also be useful for identifying negative criticisms within otherwise positive reviews.

The model performed fairly well with an overall 91.46% accuracy, though its accuracy for predicting bad reviews was much lower at 84.08%. 
This is most likely due to the fact that there are only about half as many bad review examples as good review examples. Also, no adjustments to the term weights 
was performed - as seen in the previous analysis, many indicative words from the bad review set were shared with the good review set, likely confusing the model.

Still, the model had relatively high marks, with an fscore for the good reviews of 93.64%.

The model could be further improved to increase accuracy. As discovered in the data analysis, the corpuses for both good and bad reviews share a lot of words, 
including some of the most "important" words resulting from the TF-IDF weighting. Customizing a weighting scheme that would lessen the importance of words that 
are shared between corpuses may improve accuracy as these words are more ambiguous. These words could also be deleted if they are above some threshold value on 
both DTMs.

Utilizing n-grams of at least 2 terms each may also be useful - one would expect many repeated negative word combinations in the bad review corpus (like 'not good' 
or 'not working'). As the terms are counted individually, this context is lost leading to increased positive words in the bad set and vice versa. Implementing 
n-grams would also aid in the analysis portion, providing more insight into the data.
