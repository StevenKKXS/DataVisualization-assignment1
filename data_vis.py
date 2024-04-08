import pandas as pd
import numpy as np
import os
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class IrisVisualizor:
    def __init__(self, iris_data_path) -> None:
        self.data_path = iris_data_path
        self.data = pd.read_csv(self.data_path)

    
    def density_visualization(self, save_path, x_range=(0, 8), y_range=(0, 8), is_show=False):
        # self.data['variety'] has three classes: 'Setosa', 'Versicolor', 'Virginica'
        # Extract data for each class
        setosa_data = self.data[self.data['variety'] == 'Setosa']
        versicolor_data = self.data[self.data['variety'] == 'Versicolor']
        virginica_data = self.data[self.data['variety'] == 'Virginica']

        # Create a figure with two subplots in horizontal form
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
  
        # Plot the density for sepal length and width
        sns.kdeplot(data=setosa_data, x='sepal.length', y='sepal.width', ax=axes[0], cmap='Reds', fill=True, thresh=0.05, levels=10, alpha=0.5)
        sns.kdeplot(data=versicolor_data, x='sepal.length', y='sepal.width', ax=axes[0], cmap='Blues', fill=True, thresh=0.05, levels=10, alpha=0.5)
        sns.kdeplot(data=virginica_data, x='sepal.length', y='sepal.width', ax=axes[0], cmap='Greens', fill=True, thresh=0.05, levels=10, alpha=0.5)

        sns.scatterplot(data=setosa_data, x='sepal.length', y='sepal.width', color='red', label='Setosa', ax=axes[0], s=10)
        sns.scatterplot(data=versicolor_data, x='sepal.length', y='sepal.width', color='blue', label='Versicolor', ax=axes[0], s=10)
        sns.scatterplot(data=virginica_data, x='sepal.length', y='sepal.width', color='green', label='Virginica', ax=axes[0], s=10)

        # Set the x and y axis limits for the first subplot
        axes[0].set_xlim(x_range)
        axes[0].set_ylim(y_range)

        # Set the title and labels for the first subplot
        axes[0].set_title('Sepal Length vs Sepal Width')
        axes[0].set_xlabel('Sepal Length')
        axes[0].set_ylabel('Sepal Width')

        # Plot the density for petal length and width
        sns.kdeplot(data=setosa_data, x='petal.length', y='petal.width', ax=axes[1], cmap='Reds', fill=True, thresh=0.05, levels=10, alpha=0.5)
        sns.kdeplot(data=versicolor_data, x='petal.length', y='petal.width', ax=axes[1], cmap='Blues', fill=True, thresh=0.05, levels=10, alpha=0.5)
        sns.kdeplot(data=virginica_data, x='petal.length', y='petal.width', ax=axes[1], cmap='Greens', fill=True, thresh=0.05, levels=10, alpha=0.5)

        sns.scatterplot(data=setosa_data, x='petal.length', y='petal.width', color='red', label='Setosa', ax=axes[1], s=10)
        sns.scatterplot(data=versicolor_data, x='petal.length', y='petal.width', color='blue', label='Versicolor', ax=axes[1], s=10)
        sns.scatterplot(data=virginica_data, x='petal.length', y='petal.width', color='green', label='Virginica', ax=axes[1], s=10)
        # Set the x and y axis limits for the second subplot
        axes[1].set_xlim(x_range)
        axes[1].set_ylim(y_range)

        # Set the title and labels for the second subplot
        axes[1].set_title('Petal Length vs Petal Width')
        axes[1].set_xlabel('Petal Length')
        axes[1].set_ylabel('Petal Width')

        # Add legend for the classes
        axes[1].legend(['Setosa', 'Versicolor', 'Virginica'])

        # Save the figure
        plt.savefig(save_path)
        if is_show:
            plt.show()
        plt.close()
   
    def radar_visualization(self, save_path, ax_range=(0, 8), is_show=False):
        # Extract data for each class
        setosa_data = self.data[self.data['variety'] == 'Setosa']
        versicolor_data = self.data[self.data['variety'] == 'Versicolor']
        virginica_data = self.data[self.data['variety'] == 'Virginica']

        # Create a figure with a polar projection
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # Define the angles for each dimension
        angles = np.linspace(0, 2.5 * np.pi, 5, endpoint=False)

        # Plot the radar chart for Setosa
        setosa_values = setosa_data[['sepal.width', 'petal.length', 'petal.width', 'sepal.length']].values
        for i in range(setosa_values.shape[0]):
            x = np.concatenate((setosa_values[i], [setosa_values[i][0]]))
            ax.plot(angles, x, label='Setosa', color='#ff4c4c', linewidth=1, linestyle='-', markerfacecolor='#ff4c4c')

        versicolor_values = versicolor_data[['sepal.width', 'petal.length', 'petal.width', 'sepal.length']].values
        for i in range(versicolor_values.shape[0]):
            x = np.concatenate((versicolor_values[i], [versicolor_values[i][0]]))
            ax.plot(angles, x, label='Versicolor', color='#0099e5', linewidth=1, linestyle='-', markerfacecolor='#0099e5')

        virginica_values = virginica_data[['sepal.width', 'petal.length', 'petal.width', 'sepal.length']].values
        for i in range(virginica_values.shape[0]):
            x = np.concatenate((virginica_values[i], [virginica_values[i][0]]))
            ax.plot(angles, x, label='Virginica', color='#34bf49', linewidth=1, linestyle='-', markerfacecolor='#34bf49')

        # Set the radial axis limits
        ax.set_ylim(ax_range)

        # Set the angle ticks and labels
        ax.set_xticks(angles)
        ax.set_xticklabels(['Sepal Width', 'Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width'])

        
        # Add legend
        # ax.legend()
        # ax.legend(['Setosa', 'Versicolor', 'Virginica'], loc='upper right', facecolor='white', edgecolor='black')
        # Add legend
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color='#ff4c4c'), plt.Rectangle((0, 0), 1, 1, color='#0099e5'), plt.Rectangle((0, 0), 1, 1, color='#34bf49')]
        legend_labels = ['Setosa', 'Versicolor', 'Virginica']
        plt.legend(legend_handles, legend_labels, loc='upper right', facecolor='white', edgecolor='black')

        # Save the figure
        plt.savefig(save_path)
        if is_show:
            plt.show()
        plt.close()
    
    def dimensionality_reduction_visualization(self, save_path, is_show=False):
        # Extract features and target variable
        features = self.data.drop('variety', axis=1)
        target = self.data['variety']

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(features)

        # Perform PCA dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(features)

        # Create a figure with two subplots in horizontal form
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot t-SNE result
        sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=target, ax=axes[0], palette=['#ff4c4c', '#0099e5', '#34bf49'])
        axes[0].set_title('t-SNE Dimensionality Reduction')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')

        # Plot PCA result
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=target, ax=axes[1], palette=['#ff4c4c', '#0099e5', '#34bf49'])
        axes[1].set_title('PCA Dimensionality Reduction')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')
        
        # Save the figure
        plt.savefig(save_path)
        if is_show:
            plt.show()
        plt.close()
        
    def stacked_bar_visualization(self, save_path, is_show=False):
        # Extract data for each class
        setosa_data = self.data[self.data['variety'] == 'Setosa']
        versicolor_data = self.data[self.data['variety'] == 'Versicolor']
        virginica_data = self.data[self.data['variety'] == 'Virginica']

        # Calculate the bin edges
        bin_edges = np.linspace(0, 8, 17)

        # Calculate the bin labels
        bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]

        # Calculate the histogram for each class and each feature
        setosa_width_hist, _ = np.histogram(setosa_data['sepal.width'], bins=bin_edges)
        versicolor_width_hist, _ = np.histogram(versicolor_data['sepal.width'], bins=bin_edges)
        virginica_width_hist, _ = np.histogram(virginica_data['sepal.width'], bins=bin_edges)

        setosa_length_hist, _ = np.histogram(setosa_data['sepal.length'], bins=bin_edges)
        versicolor_length_hist, _ = np.histogram(versicolor_data['sepal.length'], bins=bin_edges)
        virginica_length_hist, _ = np.histogram(virginica_data['sepal.length'], bins=bin_edges)

        # Create a figure with four subplots in a 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot the stacked bar chart for sepal width
        axes[0, 0].bar(bin_labels, setosa_width_hist, label='Setosa', color='#ff4c4c')
        axes[0, 0].bar(bin_labels, versicolor_width_hist, bottom=setosa_width_hist, label='Versicolor', color='#0099e5')
        axes[0, 0].bar(bin_labels, virginica_width_hist, bottom=setosa_width_hist+versicolor_width_hist, label='Virginica', color='#34bf49')
        axes[0, 0].set_title('Sepal Width')
        axes[0, 0].set_xlabel('Width')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot the stacked bar chart for sepal length
        axes[0, 1].bar(bin_labels, setosa_length_hist, label='Setosa', color='#ff4c4c')
        axes[0, 1].bar(bin_labels, versicolor_length_hist, bottom=setosa_length_hist, label='Versicolor', color='#0099e5')
        axes[0, 1].bar(bin_labels, virginica_length_hist, bottom=setosa_length_hist+versicolor_length_hist, label='Virginica', color='#34bf49')
        axes[0, 1].set_title('Sepal Length')
        axes[0, 1].set_xlabel('Length')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot the stacked bar chart for petal width
        axes[1, 0].bar(bin_labels, setosa_width_hist, label='Setosa', color='#ff4c4c')
        axes[1, 0].bar(bin_labels, versicolor_width_hist, bottom=setosa_width_hist, label='Versicolor', color='#0099e5')
        axes[1, 0].bar(bin_labels, virginica_width_hist, bottom=setosa_width_hist+versicolor_width_hist, label='Virginica', color='#34bf49')
        axes[1, 0].set_title('Petal Width')
        axes[1, 0].set_xlabel('Width')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot the stacked bar chart for petal length
        axes[1, 1].bar(bin_labels, setosa_length_hist, label='Setosa', color='#ff4c4c')
        axes[1, 1].bar(bin_labels, versicolor_length_hist, bottom=setosa_length_hist, label='Versicolor', color='#0099e5')
        axes[1, 1].bar(bin_labels, virginica_length_hist, bottom=setosa_length_hist+versicolor_length_hist, label='Virginica', color='#34bf49')
        axes[1, 1].set_title('Petal Length')
        axes[1, 1].set_xlabel('Length')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Save the figure
        plt.savefig(save_path)
        if is_show:
            plt.show()
        plt.close()
    
    def violinplot_visualization(self, save_path, is_show=False):
        # Create a figure with four subplots in a 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot the violin plot for sepal length
        sns.violinplot(data=self.data, x='variety', y='sepal.length', ax=axes[0, 0], palette=['#ff4c4c', '#0099e5', '#34bf49'])
        axes[0, 0].set_title('Sepal Length')
        axes[0, 0].set_xlabel('Variety')
        axes[0, 0].set_ylabel('Length')

        # Plot the violin plot for sepal width
        sns.violinplot(data=self.data, x='variety', y='sepal.width', ax=axes[0, 1], palette=['#ff4c4c', '#0099e5', '#34bf49'])
        axes[0, 1].set_title('Sepal Width')
        axes[0, 1].set_xlabel('Variety')
        axes[0, 1].set_ylabel('Width')

        # Plot the violin plot for petal length
        sns.violinplot(data=self.data, x='variety', y='petal.length', ax=axes[1, 0], palette=['#ff4c4c', '#0099e5', '#34bf49'])
        axes[1, 0].set_title('Petal Length')
        axes[1, 0].set_xlabel('Variety')
        axes[1, 0].set_ylabel('Length')

        # Plot the violin plot for petal width
        sns.violinplot(data=self.data, x='variety', y='petal.width', ax=axes[1, 1], palette=['#ff4c4c', '#0099e5', '#34bf49'])
        axes[1, 1].set_title('Petal Width')
        axes[1, 1].set_xlabel('Variety')
        axes[1, 1].set_ylabel('Width')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Save the figure
        plt.savefig(save_path)
        if is_show:
            plt.show()
        plt.close()

if __name__ == '__main__':
    iris_data_path = op.join('data', 'iris.csv')
    iris_vis = IrisVisualizor(iris_data_path)
    iris_vis.density_visualization(op.join('pics', 'Petal Density map.png'))
    iris_vis.radar_visualization(op.join('pics', 'Radar chart.png'))
    iris_vis.dimensionality_reduction_visualization(op.join('pics', 'Dimensionality Reduction.png'))
    iris_vis.stacked_bar_visualization(op.join('pics', 'Stacked Bar.png'))
    iris_vis.violinplot_visualization(op.join('pics', 'Violinplot.png'))