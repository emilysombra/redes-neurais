from perceptron import Perceptron
import pandas as pd


def main():
    df = pd.read_csv("teste1.csv")
    x = df[['x1', 'x2', 'x3']].values
    y = df['d'].values

    p = Perceptron()
    p.fit(x[:10], y[:10])
    print(p.predict(x[10:]))


if __name__ == '__main__':
    main()
