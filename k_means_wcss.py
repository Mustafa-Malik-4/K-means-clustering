
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np


def dist(u,v):
    result = (((v[0]-u[0])**2)+((v[1]-u[1])**2))**(0.5)
    return result

def init_centroid(d):
    X=[]
    Y=[]
    for b in d:
        X.append(b[0])
        Y.append(b[1])
    x = random.randrange(int(min(X)), int(max(X)))
    y = random.randrange(int(min(Y)), int(max(Y)))
    return(x,y)

def new_centroid(class_):
    X=0
    for i in class_:
        X+=i[0]
    x = X/len(class_)
    Y=0
    for i in class_:
        Y+=i[1]
    y = Y/len(class_)
    return((x,y))

def k_means_helper(data, K):
    passing = 0
    while True:
        if K > 10: return("Error: 'k_means' does not support more than 10 clusters.")
        letters = 'ABCDEFGHIJ'
        clusters = {}
        cent_list = []
        for k in range(K):
            cent_list.append(init_centroid(data))
            clusters[letters[k]] = []
        clusters_prev = []
        clusters_new = sorted((k, sorted(v)) for k, v in clusters.items())
        count = 0
        while clusters_prev != clusters_new:
            clusters_prev = sorted((k, sorted(v)) for k, v in clusters.items())
            for p in data:
                assigned_c = None
                min_dist = float('inf')
                for c in cent_list:
                    if dist(p,c) < min_dist:
                        min_dist = dist(p,c)
                        assigned_c = c
                index = letters[cent_list.index(assigned_c)]
                for key in clusters:
                    if p in clusters[key]:
                        clusters[key].remove(p)
                clusters[index].append(p)
            clusters_new = sorted((k, sorted(v)) for k, v in clusters.items())
            i = 0
            for j in clusters.values():
                if len(j) == 0:
                    print('Error: One or more clusters are empty, trying again.')
                else:
                    passing = 1
                    cent_list[i] = new_centroid(j)
                    i+=1
            count+=1
        if passing == 1:
            return [clusters, count]


def plot_points(clusters, K, count, name):
        letters = 'ABCDEFGHIJ'
        colors = ['red', 'blue', 'green', 'orange', 'purple',
                'magenta', 'cyan', 'brown', 'black', 'yellow']
        color_pairs = dict()
        c=0
        for l in letters:
            color_pairs[l] = colors[c]
            c+=1
        for key, points in clusters.items():
            x, y = zip(*points)
            plt.scatter(x, y, color=color_pairs[key], label=key)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'K-means Clustering with K = {K}'+'\n'+f'Iterations: {count}')
        plt.suptitle(f'Iterations: {count}', y=1.05, fontsize=18)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(f'output/{name}.png')
        plt.close()


def compute_wcss(clusters):
    wcss = 0
    centroids = {}
    for cluster_name, points in clusters.items():
        if len(points) == 0:
            print(f"Warning: Cluster {cluster_name} is empty. Skipping centroid calculation.")
            continue
        centroid = tuple(np.mean(points, axis=0))
        centroids[cluster_name] = centroid
        for point in points:
            wcss += np.sum((np.array(point) - np.array(centroid)) ** 2)
    return wcss


def show_elbows(wcss_dict, optimal_K, name):
    ks = list(wcss_dict.keys())
    wcss = list(wcss_dict.values())
    plt.plot(ks, wcss, marker='o')
    plt.axvline(optimal_K, color='red', linestyle='--', label=f'Optimal K = {optimal_K}')
    plt.xlabel('K')
    plt.ylabel('WCSS')
    plt.title('Elbow Plot')
    plt.legend()
    plt.savefig(f'elbow_output/{name}.png')
    plt.close()


def find_elbow(wcss_dict, name, show):
    ks = np.array(list(wcss_dict.keys()))
    wcss = np.array(list(wcss_dict.values()))
    
    line_start = np.array([ks[0], wcss[0]])
    line_end = np.array([ks[-1], wcss[-1]])
    
    distances = []
    for i in range(len(ks)):
        point = np.array([ks[i], wcss[i]])
        distance = np.abs(np.linalg.norm(np.cross(line_end - line_start, line_start - point)) / 
                          np.linalg.norm(line_end - line_start))
        distances.append(distance)
    
    elbow_index = np.argmax(distances)
    optimal_K = ks[elbow_index]
    if show==True:
        show_elbows(wcss_dict, optimal_K, name)
    return optimal_K



def k_means(data, name, plot):
    wcss_dict = {}
    clusters_dict = {}
    for K in range(1, 11):
        best_wcss = float('inf')
        best_clusters = None
        best_count = None
        for _ in range(1, 11):
            result = k_means_helper(data, K)
            clusters, count = result[0], result[1]
            current_wcss = compute_wcss(clusters)
            if current_wcss < best_wcss:
                best_wcss = current_wcss
                best_clusters = clusters
                best_count = count
        wcss_dict[K] = best_wcss
        clusters_dict[K] = (best_clusters, best_count)
    elbow = find_elbow(wcss_dict, name, show=True)
    final_clusters, final_count = clusters_dict[elbow]
    if plot == True:
        plot_points(final_clusters, elbow, final_count, name)
    else:
        return wcss_dict


def load_input(filename, X, Y):
    df = pd.read_csv(filename)
    data = []
    for row in range(0,len(df)):
        t = (df.loc[row, X], df.loc[row, Y])
        data.append(t)
    return data


if __name__ == '__main__':
    
    print("Process Started")
    
    #iris_data = load_input('data/Iris.csv','PetalLengthCm', 'PetalWidthCm')
    #k_means(iris_data, "Iris_2", plot=True)

    #mall_data = load_input('data/Mall_Customers.csv','Spending Score (1-100)', 'Annual Income (k$)')
    #k_means(mall_data, "mall_2", plot=True)

    #blob_data = load_input('data/blob.csv','x', 'y')
    #k_means(blob_data, "blob", plot=True)

    #triangle_data = load_input('data/triangle.csv','x', 'y')
    #k_means(triangle_data, "triangle", plot=True)

    #network_data = load_input('data/network.csv','x', 'y')
    #k_means(network_data, "network", plot=True)
    
    print("Process Complete")