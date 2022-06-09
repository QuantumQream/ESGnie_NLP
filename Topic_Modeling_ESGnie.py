import numpy as np
import pandas as pd
from top2vec import Top2Vec

#Load the trained model
#The model consists on 3 steps:
    #Joint word/document embedding, this process forms the semantic space (The algorithm used is Word2Vec)
    #Dimensionality reduction, similar to PCA or encoder network, this projects the document vectors into a lower dimensionality space (the algorithm is UMAP)
    #Clustering, this is the final step of the algorithm, we form clusters of documents around the centroids of the topic vectors (the algorithm is HBDscan)
    #Each cluster is then categorized as a topic
model = Top2Vec.load("ESGnie_emissions_model_V1")

#load the dropbox data
data = pd.read_csv('ESGnie_target_descriptions.csv')

#Creating a list of unique target descriptions. These were the documents we gave the model
docs = list(set(data['target_explanation_english'].tolist()))

#Create a new dataframe with the documents and add the topic labels/scores to it
df = pd.DataFrame(docs, columns=['text'])
document_topic_num = model.get_documents_topics(doc_ids=np.arange(0,6712))[0]
document_topic_score = model.get_documents_topics(doc_ids=np.arange(0,6712))[1]
df['topic_num'] = document_topic_num
df['topic_score'] = document_topic_score


#We form groups of similar topics, this containts not only methods of emission reductions but also information about the companies goals, location, etc.
conditions=[
    df['topic_num'].isin([0,32,41,72,66,47]),
    df['topic_num'].isin([1,18,52,24,58,61,68,25]),
    df['topic_num'].isin([2,10,19,69]),
    df['topic_num'].isin([3,8]),
    df['topic_num'].isin([5,17,26,73,43]),
    df['topic_num'].isin([6,39,53]),
    df['topic_num'].isin([7,48,49,51,74]),
    df['topic_num'].isin([11,67]),
    df['topic_num'].isin([12,22,56,54]),
    df['topic_num'].isin([13]),
    df['topic_num'].isin([14]),
    df['topic_num'].isin([15,70]),
    df['topic_num'].isin([16]),
    df['topic_num'].isin([23,55]),
    df['topic_num'].isin([28,20,21,30]),
    df['topic_num'].isin([29]),
    df['topic_num'].isin([34]),
    df['topic_num'].isin([36]),
    df['topic_num'].isin([38,62,31,33]),
    df['topic_num'].isin([40]),
    df['topic_num'].isin([44]),
    df['topic_num'].isin([46]),
    df['topic_num'].isin([50]),
    df['topic_num'].isin([57]),
    df['topic_num'].isin([27,37,35,4,9,42,45,59,60,63,64,65,71])
]

choices=[
    'SBTI (+IPCC) and SDA/ADA',
    'Goals for fiscal years for domestic and overseas locations (Asia)',
    'General Scope-3: Goods/Services, Supply-Chain, Commuting',
    'Net-Zero carbon emissions by 2030-2060',
    'Electricity Efficient Lighting, Air-Conditioning and Refrigerants in Work Plants',
    'Reduce Fuel Consumption (Modern Engines/Electric Vehicles)',
    'General (Unclear)',
    'Replace Coal Plants with Natural Gas, Solar and Wind Powered Systems',
    'CO2 Emission Metrics (Tons/Year)',
    'Discussion of Factories and Plants in Asia',
    'Reduce Scope1+2 Emissions in Office Spaces',
    'Purchase Electricy with Guarantees of Origin and Renewable Certificates',
    'Reduce Stationary Combustion Emissions',
    'Recycling',
    'Renewable Energy (Switch/Invest) in Solar and Wind',
    'INDC',
    'Energy Efficiency in Sold Products',
    '3% Solution, WWF and CDP',
    'General Yearly Goals',
    'Market-Based',
    'Reduce Fugitive Methane from Natural Gas Processes',
    'Gold Standard Certificates and Carbon Credits',
    'Waste/kg Calculations',
    'Reduce Flaring Emissions to Zero by 2025-2030',
    'Others',
]

df['group'] = np.select(conditions, choices, default='None')


#We only grab the groups that have to do with emission reductions
activities_df = df[
    (df['group']=='Reduce Fuel Consumption (Modern Engines/Electric Vehicles)')|
    (df['group']=='Renewable Energy (Switch/Invest) in Solar and Wind')|
    (df['group']=='General Scope-3: Goods/Services, Supply-Chain, Commuting')|
    (df['group']=='Purchase Electricy with Guarantees of Origin and Renewable Certificates')|
    (df['group']=='Replace Coal Plants with Natural Gas, Solar and Wind Powered Systems')|
    (df['group']=='Electricity Efficient Lighting, Air-Conditioning and Refrigerants in Work Plants')|
    (df['group']=='Recycling')|
    (df['group']=='Gold Standard Certificates and Carbon Credits')|
    (df['group']=='Energy Efficiency in Sold Products')|
    (df['group']=='Reduce Stationary Combustion Emissions')|
    (df['group']=='Market-Based')|
    (df['group']=='Reduce Fugitive Methane from Natural Gas Processes')|
    (df['group']=='Reduce Flaring Emissions to Zero by 2025-2030')
]

#Quick little de-tour to add one final method by extracting Decarbonization from the SBTI group
def detect_method_sbti(text):
    if 'decarbonization' in text.lower():
        return 'SDA/ADA'
    elif 'sda' in text.lower():
        return 'SDA/ADA'
    else:
        return 'N/A'
         
#Pandas, pandas, more pandas...
sbti_df = df[df['group']=='SBTI (+IPCC) and SDA/ADA']
sbti_df['method'] = sbti_df['text'].apply(detect_method_sbti)
sda_df = sbti_df[sbti_df['method']=='SDA/ADA']
complete_activities = activities_df.append(sda_df)
complete_activities.drop('method',axis=1,inplace=True)


#Save the 3 new dataframes: the original dataframe with the topic numbers and groups and the dataframe that only containts emission reduction methods
complete_activities.to_csv('complete_activities.csv')
df.to_csv('topic_groups.csv')