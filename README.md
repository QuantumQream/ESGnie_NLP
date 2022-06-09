## ESGnie_NLP: Description, Results & Prospects

# What is this?
This repository contains a **Topic Modeling Algorithm** that has been trained on the ESGnie_target_descriptions.csv file.
The model uses jointly-embedded words and document vectors, dimensionality reduction and finally clustering. But let's not get too ahead of ourselves...

We will go over the following topics:

- The problem
- How I chose this model
- The results
- Advantages & Drawbacks
- What can be done next?

# TLDR
This is a trained topic modeling algorithm that identified 75 different topics (clusters) from the corpus of documents, from these initial clusters it's possible to explore each one on its own by using the included query tools about top documents/words ocurrance and create useful Wordcloud visualizations.

After a bit of work these 75 clusters were reduced to 25 relevant groups that included emisson reduction methods, company goals, institutions followed, etc...

Finally from these 25 groups, 14 methods of emission reduction were identified that not only mention a specific measure taken but also includes useful information about the context and situation on how they were implemented

# How to run the script
- Before running the script please unzip the model.rar file and extract it to the same folder with all the other files in the repository
- A text file with all dependencies is also attached
- After running the script, two new csv files will be created containing the group/activity each document belongs to

# The problem
The tasks of identifying emission reduction methods from a large sample of documents can be considered an **unsupervised learning** problem,
that is, a problem where we want to categorize documents based on what methods of emission reduction it mentions but where we don't have the luxury of having pre-defined labels for training
> It doesn't have to be classification... Why not use NERs?

It's not a bad idea, however this approach relies on a very strong understanding of the documents. If we were to add emission reduction methods as entities and then look for them in the corpus of documents
we should first have a good idea of what methods will be mentioned and also how they will be mentioned.

For example consider a method as general as _Enery Efficiency_:
What if gets mentioned as 'installing efficient lighting in office space', or 'replace regular fuel engines with modern engines such as diesel' or 'reduce Scope1+2 emissions following SBTI guidelines'.
All of these options could potentially fit under the umbrella of _Energy Efficiency_ but it's not directly obvious how we could anticipate all the possible ways a company could be energy efficient.

Moreover the categories 'efficient lighting', 'fuel engines' and 'sbti guidelines' actually sound like very distinct topics on their own, so in my eyes it would be better to identify
**topic clusters** and then make sense of the resulting topics.

A final complication has to do with duplicate (and almost identical) documents, we will talk about this aspect later when we get to explain the results


# How I chose this model
Simple models such as LDA or NMF with tf-idf vectorizers were not too effective, having to specify an arbitrary number of topics mixed with having a very similar-looking corpus did not yield particularly good results

A better approach would be to create a **joint embedding** of both word and document vectors in a _semantic space_.
However this would also suggest that some form of **dimensionality reduction** (like PCA, or an Encoder) should take place before **clustering** in order to make this more efficient

Lucky for us, this exact process is pretty much exactly built into the library [Top2Vec](https://github.com/ddangelov/Top2Vec). First the algorithm [Word2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) creates the semantic space,
the algorithm [UMAP](https://arxiv.org/abs/1802.03426) (~~This brings back nightmares from General Relativity~~) does dimensionality reduction and finally the topic clusters are created with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan).

Having everything conveniently packed into a single library certainly helps the process


# The results
The model identified 75 topics, and classified documents by giving each one a **topic number** to which it belongs as well as a **topic score** which measures how close this document is with respect to the centroid of the cluster (cosine-similarity)
Just with this information alone, the model can do some pretty incredible things. For example we can take a look at the topic sizes:

```
>>> topic_sizes, topic_nums = model.get_topic_sizes()
>>> print(f'Topic {topic_nums[5]} has {topic_sizes[5]} documents')
  Topic 5 has 197 documents
```

Furthermore we can take a look at the most representative documents for each topic
```
>>> documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=11, num_docs=5)
    print(f'Top 5 documents for Topic {topic_nums[11]} are:')
    print()
    for doc, score, doc_id in zip(documents, document_scores, document_ids):
      print(f"Document: {doc_id}, Score: {score}")
      print("-----------")
      print(doc)
      print()
```
```
Top 5 documents for Topic 11 are:

Document: 254, Score: 0.856393039226532
-----------
We have installed 24 MW solar rooftop and 10 MW wind power. We are replacing coal with renewable biomass.

Document: 1886, Score: 0.844364583492279
-----------
Vistra has retired ~13,000 MW of coal and gas plants since 2010, contributing to a majority of the emissions reduction progress.

Document: 737, Score: 0.8438484072685242
-----------
Domestic coal-fired power plants will be gradually faded out from aging power plants, and by loading gasification equipment on existing equipment, it will be upcycled as a highly efficient power generation system using hydrogen, and emissions will be reduced by 40%.

Document: 49, Score: 0.8193049430847168
-----------
Installation of 2 new gas turbines with following shutdown of the coal-fired boilers in our power plant, savings of min 155.000 tCO2e/a

Document: 1830, Score: 0.7878034114837646
-----------
A major upgrade of the Tawke CPF is underway to improve energy management, centralize flares, as well as remove direct fired heaters.
```

We can also understand each topic by creating a WordCloud of its most representative vocabulary
```
>>> model.generate_topic_wordcloud(topic_num=11)
```
![Topic 11 WordCloud](https://github.com/QuantumQream/ESGnie_NLP/blob/main/WordCloud%20Picures/Reduce_Coal_Plants.png)

To summarize all this information I went through each topic cluster and I divided them into 25 **groups** that involve not only specific methods, but also activities particular to a portion of companies and organization guidelines

Finally from there we can extract the most relevant measures that companies are taking to reduce carbon emissions (14 activities)
![Emission Reduction Methods](https://github.com/QuantumQream/ESGnie_NLP/blob/main/WordCloud%20Picures/Emission_Reduction_Methods.png)

These results are very interesting to me, because it gives me a broader understanding and context of how each measure is being implemented, for example, I can see that the use of natural gas is relevant to replace traditional coal plants
and that it also gets mentioned often with renewable energy sources, but I can also see that there are considerable problems with accidental methane emissions.

With regards to almost identical topics, the model identified their similarities and created topics dominated by their structure, which on one hand creates a distorted classification of other documents that happen to fall on that cluster, but it also serves to clean the rest of the topics by grouping similar documents together.
The devil is in the details here, so further improvements must be implemented with care and clear objectives in mind.

# Advantages
- Light and easy to implement
- Performs very well at identifying topics on its own
- Not only tells us about particular methods but also how these methods are being implemented in different contexts
- Useful query tools to find relevant documents for each topic, look for semantically similar words/documents and create automatic hierarchical groupings (~~which I didn't use this time, but it can be powerful if well trained~~)
- Feature to add more documents on already trained model
- Excellent Wordcloud visualizations

# Disadvantages
- Not diretly replicable since the model is based on stochastic algorithms, training the model from scratch will result in slightly different ording and sizes for topics (However we can save an existing model and load it to guarantee consistency)
- A bit difficult to change the structure of the model since it all happens under the hood. Nevertheless by knowing each step of the process and getting in touch with each library it's perfectly possible to fine tune the model

# What can be done from here?
Being an unsupervised learning problem, it's expected that at some point a bit of manual labor is necessary to understand the results, however this can be a great asset as well.
If we were to be content with the results, the further groupings and activities identified, the performance of the clustering process, etc.
It would mean that we now have a good training data set from which laters documents could be compared against.

In a way being content with this project means that later down the problem could be reduced to a simple supervised multi-class classification problem. A model can be trained to just choose from our predefined labels

Further improvements can be done by analizing each manually (or automatically) created group, and perform similar clustering of them to extract more detailed information

Further analysis can be made by insvestigating which organizations or institutions are mentioned, and by investigating their guidelines further improve the identification process for specific methods of emission reduction



