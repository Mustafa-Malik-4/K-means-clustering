
import matplotlib.pyplot as plt
import random
import pandas as pd


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

def k_means(data, K, name, plot):
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

        if plot == True and passing == 1:
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
            return
        if plot == False and passing == 1:
            return(clusters)


def load_input(filename, X, Y):
    df = pd.read_csv(filename)
    data = []
    for row in range(0,len(df)):
        t = (df.loc[row, X], df.loc[row, Y])
        data.append(t)
    return data



data1= [(12, 1), (11, 2), (9, 3), (8, 1), (10, 2), (11, 4), (8, 3),
          (9, 12), (11, 10), (10, 8), (8, 9), (12, 11), (10, 10), (9, 8),
          (16, 4), (14, 5), (15, 7), (17, 6), (14, 8), (15, 6)]



if __name__ == '__main__':
    
    print("Process Started")
    
    #k_means(data1, 3, "data", plot=True)

    #spotify_data = load_input('data/spotify.csv', 'track_popularity', 'tempo')
    #k_means(spotify_data, 3, "Spotify", plot=True)
        
    #iris_data = load_input('data/Iris.csv','PetalLengthCm', 'PetalWidthCm')
    #k_means(iris_data, 3, "Iris", plot=True)

    #mall_data = load_input('data/Mall_Customers.csv','Spending Score (1-100)', 'Annual Income (k$)')
    #k_means(mall_data, 5, "mall", plot=True)


    #network_data = load_input('data/network.csv','x', 'y')
    #k_means(network_data, 5, "network", plot=True)
    
    print("Process Complete")






"""


"""