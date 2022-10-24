# Analyzing Customer Reviews from Amazon

## Introduction

Customer reviews are a very valuable tool for businesses as they provide feedback for the business to improve to increase customer satisfaction. Often, these reviews are on some scale (like from 1-10); obviously, a high score is great for the business while a low score is bad. However, to truly understand these reviews, one must go through them and determine exactly what is being praised or criticized. The goal of this project was to create a model that could predict whether online reviews for a product are positive or negative based on the text content alone. The ability to distinguish between good and bad reviews would allow for a more accurate classification of customer sentiment.

This project was originally done in the RStudio environment, but I moved the code into a JuPyterNotebook environment for easier viewing of the outputs.

### Business Task

Originally, the business task for this project was to build a model which could identify whether a review for headphones from Amazon is good or bad, depending on the text within the review. This data could be used to guide product development and quality control efforts to increase customer satisfaction with the business's products. However, I decided that the scope of that task was too large and decided to split the project into two sections - analyzing the review content and finding the most important words from either good or bad reviews, and building a model that could identify good or bad reviews.

## Dataset

The [dataset](https://www.kaggle.com/shitalkat/amazonearphonesreviews) used to create the model is called “Amazon Earphones Reviews” and was posted by user Shital Kat on Kaggle. It is a 2.5MB csv file that consists of 14,337 reviews and 4 columns:

* Title of review
* Body of review
* Rating (1-5)
* Product name

<p align="center">
    <img src="images\review_text_df.PNG" height="200"><br>
    <em>A couple of example rows from the dataset</em>
</p>
<br>

Instead of building a model to predict a rating from 1-5 stars, to simplify the analysis, I grouped the reviews into "good" and "bad" categories. The "good" category would contain 4-5 star reviews, while the "bad" category would contain 1-3 star reviews. From a business perspective, it might not make much sense to categorize the data into too many categories, especially if customer satisfaction is a top priority - any rating lower than perfect can be considered bad as it suggests the product needs improvement. From a modeler's perspective, having too many unnecessary classes can confuse the model. 

Reducing the number of classes also addresses one of the big issues with the dataset - the data is heavily skewed toward good reviews. In fact, the 4-5 star category makes up 2/3 of the data entirely. While binning the reviews does not equalize the number in either category, it improves the situation greatly.

<p align="center">
    <img src="images\pie_review_split.png" height="400">
    <img src="images\pie_good_bad.png" height="400"><br>
    <em><b>Left</b>: The distribution of 1-5 star reviews from the original dataset<br>
    <b>Right</b>: The distribution of the grouped good and bad reviews</em>
</p>
<br>

## Packages used

>*tm*: Text mining package to build the document-term matrices (DTM)<br>
>*SnowballC*: Contains the base stopwords dictionary<br>
>*ggplot2*: For data visualization tools<br>
>*wordcloud*: For building wordclouds<br>
>*RColorBrewer*: For additional color palettes for plots<br>
>*e1071*: For building SVM prediction models

# Analysis

## Buliding the DTMs

To build the Document-Term Matrix, I performed the following steps to build a dictionary of tokens from the dataset:

* Initialize the corpus 
* Remove punctuation (in particular, commas, periods, and slashes)
* Remove numbers
* Convert all words to lower case
* Remove stopwords 
* Convert any applicable words to their stems
* Eliminate white spaces from the words
* Eliminate sparse words
    * For the overall analysis, this includes words that appear in fewer than 2% of all reviews
    * For all other analyses, this included words that appear in less than some number of documents (low document frequency)

I used TF (term-frequency) weighting to determine the most important words in the dataset. TF essentially counts the number of instances of a given token that appear in a given document, and I will use word frequencies to determine the likelihood of a review being good or bad.

The example review text below shows the evolution from the original text to the preprocessed text to the DTM representation:

<p align="center">
    <img src="images\data_rep.PNG" height="500"><br>
</p>
<br>

## Visualizing the Most Important Words Overall

I plotted the 30 most important words extracted from the review text using the TF-IDF weighting to visualize the results. I also created a word cloud to emphasize the words themselves

<p align="center">
    <img src="images\tf_barh.PNG" height="600"><br>
    <img src="images\tf_cloud.PNG" height="600">
</p>
<br>

Besides words referring to sentiment (like good, better), a few words regarding headphone quality appear to be the most discussed across all reviews. Some conclusions can be made, but for a more accurate analysis, the context of these words would need to be considered:<br><br>

### Most important word:
* product: Most likely referring to the product being reviewed, but could indicate other products being compared

**Various terms indicate focus on build quality:**
* qualiti/work
* batteri/life
* bluetooth/connect

**Various terms indicate focus on sound quality:**
* sound/nois
* bass/music

**Various terms indicate focus on product features:**
* bluetooth
* batteri/life
* wire/call/rang

**Various terms indicate focus on customer experience:**
* use/work/buy

**Various terms indicate focus on price and product value:**
* price/buy
* cancel

## Visualizing the Most Important Words in Good/Bad Reviews

I plotted the most common words in both the good and bad review sets to gain some insight:

<p align="center">
    <img src="images\good_barh.PNG" height="525">
    <img src="images\bad_barh.PNG" height="525">
</p>
<br>

Most notably, there are a lot of similarities between the two sets. For one, the top 4 tokens in both the good and bad review datasets are "good", "sound", "qualiti", and "product" (though with different frequencies). There are also several other shared tokens, including "batteri", "money", and "use". I believe this is partly due to my arbitrary categorization of 1-3 star reviews as "bad" and 4-5 star reviews as "good." In reality, as these ratings are on a scale, there is no binary separation. Also, given that the star-rating is set by the reviewer, the rating itself is subjective.

Another interesting thing to note is that there are more words in the bad review set (142) than the good review set (115), despite there being about twice as many good reviews. I believe there are two main reasons for this. First, I removed tokens from both datasets that aren't present in at least 2% of documents. Second, I believe bad reviews tend to be more comprehensive than good reviews. Both go hand-in-hand - if bad reviews are more detailed than good reviews, then a larger corpus would be extracted since there are more words and variation in the dataset. 

I also noticed that many good reviews focused a lot more on customer sentiment, while many bad reviews focused on issues with the product itself. A few examples are shown below:

### **Example Good Reviews**

>'Just go for it\n **Awesome best budget** wireless earphones\n'

>'**SUPERB**\n **Sound quality is good**. noise cancellation is not upto the mark. **battery quality is decent**. by charging it once you can enjoy upto 5hrs of music without any issue\n'

### **Example Bad Reviews**

>'Honest review of an edm music lover\n No doubt it has a great bass and to a great extent noise cancellation and decent sound clarity and mindblowing battery but the following **dissapointed** me though i tried a lot to adjust.1.**Bluetooth range not more than 10m2. Pain in ear due the conical buds(can be removed)3. Wires are a bit long which makes it odd in front.4. No pouch provided.5. Worst part is very low quality and distoring mic**. Other person keeps complaining about my voice.\n'

>'Review update - **Product failed after a few months**\n Update: Sadly this is the **second Boat product to conk off after few months of use**. I guess this is the end of the journey for you and I Boat Audio, **not buying any more products from your brand**. Changed the rating to reflect the same.I was in the market for a cheap pair of Bluetooth earphones and I chanced upon this on a lightning deal and bought it. I have used Boat products before, they have good build quality and their sound is alright. I’m writing this after a month or so, enough time to have lived with it and get to know quirks and other problems.• **The sound is nothing great**, it’s a bit on the bassy side. If you’re a bass lover, this is definitely a good buy.• Connectivity is quick and painless – and it works without hassle every time. **The quality of the connectivity depends and is sometimes weird**. I can put the phone on charge and use it for a distance of about 3 meters, no worries. But sometimes when I’m walking really fast or jogging there are breaks and crackles in audio. Not sure if it’s the material of my gym wear, but it works well when the phone’s in my jeans.• **Battery life is nothing to write home about** – I listen to music constantly even when I’m at work. My demands are probably too high for this price range.• It is sweat-proof as advertised. No interruptions under a light drizzle.• It’s got great build quality for the price – looks and feels premium. Doesn’t tangle at all.• It stays in your ear regardless of the intensity of the exercise which is great.On the whole a great buy if you are looking for a cheaper alternative. Wait for a sale before you get it though.Like my reviews? Please do click the helpful button. It encourages me to share more about the products I use. Thanks!\n'

<br>

## Adjusting the Weighting

As there are many shared tokens between the datasets, it's hard to determine any strong indicators of good or bad reviews since each token is analyzed without context. I wanted to focus on tokens that were more clearly connected with good or bad reviews.

To achieve this, I manipulated the document term matrices using 3 metrics to amplify the weight for terms that are more common in either good or bad reviews:

1. Total term frequency (TTF)
    * The total number of appearances in the DTM for each term
2. Average term frequency (ATF)
    * The average number of appearance for each term per document
3. Document frequency (DF)
    * The number of documents in which each term appears at least once

For a term to be weighted highly for a review set, it should have a high scores in all 3 metrics in one review set and lower scores in all 3 metrics in the other review set. I created another metric that combines these metrics that would weight the variables, which I called TFDF (Term Frequency - Document Frequency)

For each term in both the good and bad DTMs I calculated its weight with the following formulas:

$$TFDF\_Weight_{good} =  \frac{TTF_{good} * ATF_{good} * DF_{good}}{nrow(DTM_{good}) * TTF_{bad} * ATF_{bad} * DF_{bad}} $$
$$TFDF\_Weight_{bad} =  \frac{TTF_{bad} * ATF_{bad} * DF_{bad}}{nrow(DTM_{bad}) * TTF_{good} * ATF_{good} * DF_{good}} $$

To avoid division by zero , I initialized the default value for each variable as 1 (even if a given term is not present in one of the DTMs). I then scaled each weight set from 0 to 1 to reduce the skew toward the most frequent terms. The resulting weight set still had a massive skew, so I saved a version that I first applied a log transform to.

I then plotted the most important words after the weights for all shared tokens were adjusted:

<br><br>
<p align="center">
    <b><font size = "+2">Original Weights</font></b><br>
    <img src="images\good_barh.PNG" height="500">
    <img src="images\bad_barh.PNG" height="500"><br><br>
    <b><font size = "+2">Adjusted Weights</font></b><br>
    <img src="images\adjusted_good_barh.PNG" height="525">
    <img src="images\adjusted_bad_barh.PNG" height="525"><br><br>
    <b><font size = "+2">Adjusted Weights With Log Transformation</font></b><br>
    <img src="images\adjusted_scaled_good_barh.PNG" height="525">
    <img src="images\adjusted_scaled_bad_barh.PNG" height="525">
</p>
<br>


The resulting most important words in either set now feel more definitive - the top words in the good set are mostly words of positive sentiment while the top words in the bad set are split between criticisms of the product and words of negative sentiment.

I also created word clouds to emphasize these words in their respective sets. For each word cloud, the blue words are more common in the respective DTM, while the red words are more common in the other DTM 

<br>
<p align="center">
    <b><font size = "+2">Most Important Words From the Good Dataset</font></b><br>
    <img src="images\good_cloud.PNG" height="500">
    <img src="images\good_adjusted_cloud.PNG" height="500"><br>
    <em><b>Left</b>: Original Distribution<br><b>Right</b>: Updated Distribution</em><br><br>
    <b><font size = "+2">Most Important Words From the Bad Dataset</font></b><br>
    <img src="images\bad_cloud.PNG" height="500">
    <img src="images\bad_adjusted_cloud.PNG" height="500"><br>
    <em><b>Left</b>: Original Distribution<br><b>Right</b>: Updated Distribution</em><br>
    <em><b></b></em>
</p>
<br>

# Conclusion

By implementing this custom TF-DF weighting scheme, I was able to amplify the weights for terms that are more relevant to either DTM. The resulting top words from either DTM are very indicative of either good or bad reviews. In the next part of this project, I'll explore whether this weighting scheme can increase the accuracy of predictive models.

<br>
<p align="center">
    <b><font size = "+2">Top 10 words from both datasets</font></b><br>
    <img src="images\good_top10.PNG" height="350"; padding-right: 30px>
    <img src="images\bad_top10.PNG" height="350"><br>
    <em><b>Left</b>: Good DTM<br>
    <b>Right</b>: Bad DTM</em>
</p>
<br>
