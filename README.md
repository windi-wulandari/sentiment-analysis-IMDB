<img src="https://drive.google.com/uc?id=1g4t3dYxccZsuzx0xXx6saaRWvGCXGi6G" alt="My Image" width="1000" height="250">


---

# **Directory Overview**

1. **[dataset](https://github.com/windi-wulandari/sentiment-analysis-IMDB/tree/main/dataset)**  
   This directory contains the dataset used in the project, formatted as a CSV file for sentiment analysis.

2. **[project_proposal](https://github.com/windi-wulandari/sentiment-analysis-IMDB/tree/main/project_proposal)**  
   This folder contains the project proposal files in PDF format, available in two versions: Indonesian and English. The proposal covers the entire project from summary, problem statement, objectives, to conclusions, and discusses the basic theories related to models and business simulations.

3. **[requirements.txt](https://github.com/windi-wulandari/sentiment-analysis-IMDB/blob/main/requirements.txt)**  
   This file lists the commands for installing the libraries needed throughout the project, including the versions of the libraries used.

4. **[ml_pipeline](https://github.com/windi-wulandari/sentiment-analysis-IMDB/tree/main/ml_pipeline)**  
   There are two versions of the `.ipynb` notebooks in this folder, one in Indonesian and the other in English. The notebooks contain the entire pipeline from preprocessing, data exploration, to modeling. Each step is explained in detail through comments in the code using the `#` symbol and through Markdown blocks.

5. **[resources_gdrive.txt](https://github.com/windi-wulandari/sentiment-analysis-IMDB/blob/main/resources_gdrive.txt)**  
   This text file contains Google Drive links that can be accessed to download all project resources such as project proposals, notebooks, and others. This link serves as an alternative if the files on GitHub are inaccessible or do not display properly.

6. **[bow_vs_tf-idf](https://github.com/windi-wulandari/sentiment-analysis-IMDB/tree/main/bow_vs_tf-idf)**  
   This notebook explains how the Bag of Words (BoW) and TF-IDF models work and facilitates understanding the concepts by integrating them with the theories discussed in the project proposal.

7. **README**  
   This file helps users understand the project structure and how to run or use the available files.

---

# **Background of the Problem**

The modern film industry is greatly influenced by audience reviews and sentiments on platforms such as IMDb. With the massive volume of data, manual analysis becomes inefficient, making machine learning approaches a better solution. Simple models like Logistic Regression and Naive Bayes using Bag of Words and TF-IDF techniques can perform analysis quickly, efficiently, and cost-effectively. This sentiment analysis provides significant benefits for film producers, ranging from predicting success to developing more targeted marketing strategies. This step serves as an important foundation before moving on to more complex stages, such as model deployment or using more sophisticated models to further enhance accuracy and efficiency.

**Note:**  
A more complete version of the background can be accessed in the **[project_proposal](https://drive.google.com/drive/folders/1yCuCNeg-R0YPHWZ8fDguC_xmBxaYE93b?usp=sharing)** file, pages 3-6.

---

# **Objectives**

**Sentiment Analysis: Optimizing Time Savings and Cost Efficiency in the Film Industry**

The main objective of this project is to leverage machine learning technology to optimize the sentiment analysis of movie reviews. This research focuses on achieving significant improvements in time savings and cost efficiency. Thus, this research is expected to provide a more efficient solution for handling a large volume of reviews while offering a more economical and effective approach to sentiment analysis compared to traditional methods.

---

# **Specific Objectives**

1. Optimize the speed of sentiment analysis to enhance responsiveness to audience feedback.
2. Improve cost efficiency in the process of analyzing film reviews for better resource allocation.
3. Maintain high accuracy and reliability in sentiment classification to support accurate decision-making in the film industry.

---

# **Metrics**
**Business Metrics**

1. Time Savings: Time savings refers to the reduction in time required to analyze movie reviews using machine learning compared to manual methods.
2. Cost Efficiency: Cost efficiency measures the reduction in operational costs achieved through the use of machine learning models in the sentiment analysis process.

**Model Metrics**

1. Throughput/Runtime: This metric measures the number of reviews that can be processed within a certain timeframe, providing a direct indication of analysis speed. Throughput is crucial for achieving significant time analysis reduction targets.
2. Accuracy: This metric is used to measure how well the model classifies reviews overall, whether using the Bag of Words approach or TF-IDF. Accuracy provides an overview of how reliably the model predicts positive and negative sentiments.

**Note:**  
A complete version of the metrics with formulas and explanations can be accessed in the **[project_proposal](https://drive.google.com/drive/folders/1yCuCNeg-R0YPHWZ8fDguC_xmBxaYE93b?usp=sharing)** file, pages 8-9.

---

# **About the Dataset**

The IMDb dataset consists of 50,000 rows of data containing two main columns, namely "review," which contains the text of English movie reviews, and "sentiment," which indicates the sentiment label as positive (1) or negative (0). Here are the detailed explanations.

**Table 1. Structure of the IMDb Dataset**

| No. | Column     | Data Type       | Description                                                          |
|-----|-----------|-----------------|--------------------------------------------------------------------|
| 1.  | review    | String/Object   | Text of English movie reviews                                       |
| 2.  | sentiment | Integer         | Target column indicating the sentiment label as positive (1) or negative (0) based on the review |

<br>

---

# **EDA & Pre-Processing**

**Table 2. EDA & Pre-Processing**

| No. | Step - Step                                    | Description & Findings                                                                                                                  |
|-----|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| 1.  | Check for Missing Values                        | No missing values found in the dataset.                                                                                                |
| 2.  | Check for Duplicates                            | There are 418 duplicate rows; however, they were not removed as their removal decreased model performance.                           |
| 3.  | Feature Engineering                            |                                                                                                                                          |
|     | - review_length                                | This column contains the length of the review text in each row. The purpose is to analyze whether the length of the review correlates with sentiment. |
|     | - review_length_binned                         | Binning the length of reviews into several categories: 'Short', 'Medium', 'Long', 'Very Long', and 'Extreme' for further analysis.   |
| 4.  | Check Sentiment Class Distribution              | The class distribution between positive sentiment (1) and negative (0) is balanced.                                                  |
| 5.  | Check Review Length Distribution                | Data distribution is right-skewed, with the majority of reviews having lengths of 100-400 words. Outliers were not removed as their influence on the model was not significant. |
| 6.  | Check Correlation Between Review Length and Sentiment | There is a significant difference in review lengths between positive and negative sentiments, but a small Cohen's d value indicates the practical impact is not significant. |
| 7.  | Removing HTML Strips and Noise Text            | HTML elements and noise text were removed from reviews.                                                                              |
| 8.  | Removing Special Characters                     | Special characters were removed to clean the review texts.                                                                            |
| 9.  | Text Stemming                                  | Stemming was performed on the text to convert words to their base forms (e.g., "running" to "run").                                  |
| 10. | Removing Stopwords                             | Stopwords were not removed as experiments showed better model performance without removing stopwords.                                  |
| 11. | Removed Features                               | Engineered features were removed as they did not significantly impact model performance.                                              |
| 12. | Labeling the Sentiment Text                    | Labeling the sentiment column as 1 (positive) and 0 (negative).                                                                     |
| 13. | Split Data                                     | Data was split 70% for training and 30% for testing.                                                                                 |
| 14. | Term Frequency-Inverse Document Frequency (TF-IDF) | Applying TF-IDF to convert review texts into numerical vectors while considering the relative frequency of words within the reviews.   |
| 15. | Bag of Words Model                             | Using the Bag of Words model to count the frequency of word occurrences without considering the order.                                 |

**Note:**  
A complete version of the EDA & Pre-Processing with explanations for each code can be accessed in the **[ml_pipeline](https://github.com/windi-wulandari/sentiment-analysis-IMDB/tree/main/ml_pipeline)** notebook.

---

Here’s the translation of your text into English:

---

# **Fundamental Theory of Bag of Words and Term Frequency-Inverse Document Frequency (TF-IDF)**

**Bag of Words Model**
Bag of Words, often abbreviated as BoW, is one of the simplest feature extraction methods in Natural Language Processing (NLP). This technique converts textual data into vectors that can be processed by computers. The concept of a bag of words can be likened to a bag containing a collection of words from a text document. In this bag, we do not pay attention to the order or context of the occurrence of these words. The main focus is on which words appear and how often each word appears. This is why this technique is called "Bag of Words," as it is like looking at the contents of a bag, where what matters is what is inside, not the order of the objects.

**Term Frequency-Inverse Document Frequency (TF-IDF) Model**
TF-IDF (Term Frequency — Inverse Document Frequency) is a method used in text processing to weight words based on how important they are in a document or a whole set of documents (corpus). Its main purpose is to identify words that are more informative and significant compared to commonly occurring words that provide less information.

**Note:**
A more comprehensive theoretical version can be accessed in the **[project_proposal](https://drive.google.com/drive/folders/1yCuCNeg-R0YPHWZ8fDguC_xmBxaYE93b?usp=sharing)** file on pages 12-14 and the **[notebook bow_vs_tf-df](https://github.com/windi-wulandari/sentiment-analysis-IMDB/tree/main/bow_vs_tf-idf)** that provides a simpler understanding of the features produced by both model techniques.

---

# **Modeling**

1. **Best Model:** Naive Bayes and Logistic Regression show balanced and better performance compared to SVM. However, Logistic Regression slightly outperforms in terms of True Positive (TF-IDF) with 5747 TP compared to Naive Bayes in both BOW and TF-IDF and has fewer False Negatives (FN). Additionally, the execution time of all three models is relatively fast, so there are no issues related to runtime implementation.

2. **Decision:** The main focus is on the balance and effectiveness in detecting the positive class; Logistic Regression TF-IDF is the best choice among the three models.

**Note:**
A more comprehensive comparison including accuracy, runtime, and confusion matrix can be accessed in the **[project_proposal](https://drive.google.com/drive/folders/1yCuCNeg-R0YPHWZ8fDguC_xmBxaYE93b?usp=sharing)** file on pages 15-16 and **[ml_pipeline](https://github.com/windi-wulandari/sentiment-analysis-IMDB/tree/main/ml_pipeline)**.

---

# **Impact on Business & Results**

This sentiment analysis project in the film industry has achieved significant success by utilizing machine learning techniques to understand public sentiment towards movie reviews. Focusing on two main business matrices, namely Time Saving and Cost Efficiency, has yielded very satisfactory results. The results show that time savings in the analysis process can be optimized up to 99%, while cost efficiency also reaches 99%. This achievement indicates that with the implementation of the right techniques, the project not only meets but also exceeds the established targets, making a meaningful contribution to the film industry in understanding and responding to audience needs and preferences. This success opens up opportunities for further application of similar analysis methods in various other business contexts.

**Note:**
A complete business simulation version with detailed calculations can be accessed in the **[project_proposal](https://drive.google.com/drive/folders/1yCuCNeg-R0YPHWZ8fDguC_xmBxaYE93b?usp=sharing)** file on page 16.

---

# **Limitations**

One of the limitations of this project is inadequate computational resources, which has resulted in some techniques and experiments not being able to be run optimally. Additionally, the machine learning model developed has not been followed up with a deployment process, so it cannot yet be integrated into a broader recommendation system.

---

# **Next Steps**

To enhance the results and performance of the model, the next steps will involve applying more advanced deep learning approach experiments, as well as collecting and processing higher quality and more diverse data. Additionally, exploration of new techniques in Natural Language Processing (NLP) will be conducted and consideration will be given to integrating the model into a more comprehensive recommendation system to provide more accurate and relevant recommendations for users.

--- 

# **References**

- Chen, Y., Zhang, H., Liu, R., Ye, Z., & Lin, J. (2019). Experimental explorations on short text topic mining between LDA and NMF based Schemes. Knowledge-Based Systems, 163, 1-13.

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171-4186.

- Garcia, D., & Ráez, A. (2019). Coping with the long tail: Hybrid approaches to manage and analyze big text collections. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 3215-3216.

- Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 328-339.

- IMDb. (2024). About IMDb. Retrieved from https://www.imdb.com/about/

- Kim, S. M., & Kim, H. J. (2018). Sentiment classification of movie reviews using feature selection based on dynamic λ-measure. Applied Intelligence, 48(5), 1268-1285.

- Liu, B. (2020). Sentiment analysis: Mining opinions, sentiments, and emotions. Cambridge University Press.

- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692.

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies, 142-150.

- Rina. (2024). Mengenal Bag of Words pada Model NLP. Medium. Retrieved from https://esairina.medium.com/mengenal-bag-of-words-pada-model-nlp-4013ec879e26

- Rina. (2024). Mengenal Term Frequency-Inverse Document Frequency (TF-IDF) pada Model NLP. Medium. https://esairina.medium.com/mengenal-term-frequency-inverse-document-frequency-tf-idf-pada-model-nlp-e0cc571f7e37

- Smith, J., & Lee, K. (2020). Personalized content recommendation in streaming platforms using deep learning-based sentiment analysis. IEEE Transactions on Multimedia, 22(3), 625-637.

- Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune BERT for text classification?. In China National Conference on Chinese Computational Linguistics (pp. 194-206). Springer, Cham.

- Wang, Y., Wang, M., & Xu, W. (2021). A sentiment-enhanced hybrid recommender system for movie recommendation: A big data analytics framework. Wireless Communications and Mobile Computing, 2021.

- Yu, X., Liu, Y., Huang, X., & An, A. (2020). Mining online reviews for predicting sales performance: A case study in the movie domain. IEEE Transactions on Knowledge and Data Engineering, 24(4), 720-734.

- Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8(4), e1253.

---
Kaggle inspiration click **[here](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/comments)**

