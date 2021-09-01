import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from config import Constants, Hyper

class Chart:
    def show_country_distribution(df):
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (20,10)

        # Plot the number of tokens of each length.
        sns.countplot(y=df["Country"], order = df['Country'].value_counts().index)
        plt.title('Country Distribution')
        plt.xlabel('# of Tweets')
        plt.ylabel('')
        chart = Chart.get_graph_file(Constants.country_distribution_graph)
        plt.savefig(chart) 
        plt.close()         # Use close to ensure plt is reset for future use
    
    def show_sentiment_distribution(df):
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (20,10)

        # Plot the number of tokens of each length.
        sns.countplot(y=df["sentiment"], order = df["sentiment"].value_counts().index)
        plt.title('Sentiment Distribution')
        plt.xlabel('# of Tweets')
        plt.ylabel('')
        chart = Chart.get_graph_file(Constants.sentiment_distribution_graph)
        plt.savefig(chart) 
        plt.close()         # Use close to ensure plt is reset for future use
    
    def show_combined_distribution(df):
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (20,10)

        # Plot the number of tokens of each length.
        sns.countplot(y=df, order = df.value_counts().index)
        plt.title('Combined Country/Sentiment Distribution')
        plt.xlabel('# of Tweets')
        plt.ylabel('')
        chart = Chart.get_graph_file(Constants.combined_distribution_graph)
        plt.savefig(chart) 
        plt.close()         # Use close to ensure plt is reset for future use
                    
    def show_tokens_per_tweet(token_lengths):
        # print graph of tweet token lengths
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (10,6)

        # Truncate any tweet lengths greater than 128.
        lengths = [min(l, Constants.tokens_max_length) for l in token_lengths]

        # Plot the distribution of tweet lengths.
        sns.distplot(lengths, kde=False, rug=False)

        plt.title('Tweet Lengths')
        plt.xlabel('Tweet Length')
        plt.ylabel('# of Tweets') 
        chart = Chart.get_graph_file(Constants.Tweet_length_graph)
        plt.savefig(chart)
        plt.close()         # Use close to ensure plt is reset for future use

    def show_training_stats(df_stats):
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (15,10) 
        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4]) 
        chart = Chart.get_graph_file(Constants.training_validation_loss_graph)
        plt.savefig(chart)   
        plt.close()         # Use close to ensure plt is reset for future use 
        
    def show_wordcloud(wordcloud, country):
        filename = f"wordcloud_{country}.png"
        wordcloudfig = os.path.join(Constants.images_dir, filename)
        plt.title(f"Word cloud for {country}")
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(wordcloudfig)
        plt.close()
        
    def show_confusion_matrix(matrix, combined_list):
        labels = Chart.get_labels()
        countries = ", ".join(Hyper.selected_countries)
        title = f"Confusion matrix for the countries: {countries}\n"
        x = int(round(4 * Hyper.num_labels))
        y = int(round(x * 0.5))
        Chart.make_confusion_matrix(matrix, group_names=labels, categories=combined_list, figsize=(x, y), title=title)

    def get_labels():
        labels = []
        TP = "TP"
        TN = "TN"
        FP = "FP"
        FN = "FN"
        for i in range(Hyper.num_labels):
            for j in range(Hyper.num_labels):
                if i == j:
                    if i % 2 == 0:
                        labels.append(TN)
                    else:
                        labels.append(TP)
                else:
                    if i > j:
                        labels.append(FP)
                    else:
                        labels.append(FN)
                        
        return labels
        
    ''' Code below taken frm https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py '''    
    def make_confusion_matrix(cf, 
                                group_names=None,
                                categories='auto',
                                count=True,
                                percent=True,
                                cbar=True,
                                xyticks=True,
                                xyplotlabels=True,
                                sum_stats=True,
                                figsize=None,
                                cmap='Blues',
                                title=None):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                    Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                    See http://matplotlib.org/examples/color/colormaps_reference.html
                    
        title:         Title for the heatmap. Default is None.
        '''


        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names)==cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(cf) / float(np.sum(cf))

            #if it is a binary confusion matrix, show some more stats
            if len(cf)==2:
                #Metrics for Binary Confusion Matrices
                precision = cf[1,1] / sum(cf[:,1])
                recall    = cf[1,1] / sum(cf[1,:])
                f1_score  = 2*precision*recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy,precision,recall,f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""


        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize==None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False


        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)
        
        if title:
            plt.title(title)
            
        chart = Chart.get_graph_file(Constants.confususion_matrix_graph)
        plt.savefig(chart)   
        plt.close()         # Use close to ensure plt is reset for future use 
        
    def get_graph_file(file):
        return os.path.join(Constants.images_dir, file)