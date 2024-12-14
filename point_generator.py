
import random
import pandas as pd

def generate_clustered_points(num_points, spread=1.0, bounds=(-1000, 1000)):
    points = []
    centers = [
        (random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1]))
        for _ in range(1)
    ]
    for _ in range(num_points):
        center = random.choice(centers)
        point = (
            random.gauss(center[0], spread),
            random.gauss(center[1], spread)
        )
        points.append(point)
    return points

if __name__ == '__main__':
    points = generate_clustered_points(1000)
    X_values = []
    Y_values = []
    for point in points:
        X_values.append(point[0])
        Y_values.append(point[1])
    df = pd.DataFrame({'X':X_values, 'Y':Y_values})
    # Write the DataFrame to a CSV file
    df.to_csv('data/gpt_points.csv', index=False) 


