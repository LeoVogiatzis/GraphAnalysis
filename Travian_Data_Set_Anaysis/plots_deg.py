def diff_visualizations(g_directed):
    # nx.draw(g_directed)
    nx.draw_random(g_directed)
    # nx.draw_spectral(g_directed)
    plt.savefig('graph3.png')


def dashb():

    app = dash.Dash()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(10), [i ** 2 for i in range(10)])
    ax.grid(True)
    plotly_fig = mpl_to_plotly(fig)

    app.layout = html.Div([
        dcc.Graph(id='matplotlib-graph', figure=plotly_fig)

    ])

    app.run_server(debug=True, port=8010, host='localhost')


def vis3(g_directed):
    bet_cen = nx.betweenness_centrality(g_directed)
    # Create ordered tuple of centrality data
    cent_items = [(b, a) for (a, b) in bet_cen.iteritems()]
    cent_items.sort()
    cent_items.reverse()

    # Closeness centrality
    clo_cen = nx.closeness_centrality(g_directed)
    cent_items2 = [(b, a) for (a, b) in clo_cen.iteritems()]
    cent_items2.sort()
    cent_items2.reverse()

    # Create figure and drawing axis
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    # Create items and extract centralities
    items1 = sorted(cent_items.items())
    items2 = sorted(cent_items2.items())
    xdata = [b for a, b in items1]
    ydata = [b for a, b in items2]
    # Add each actor to the plot by ID
    for p in range(len(items1)):
        ax1.text(x=xdata[p], y=ydata[p], s=str(items1[p][0]), color="b")

    # use NumPy to calculate the best fit
    slope, yint = plt.polyfit(xdata, ydata, 1)
    xline = plt.xticks()[0]
    yline = map(lambda x: slope * x + yint, xline)
    ax1.plot(xline, yline, ls='--', color='b')
    # Set new x- and y-axis limits
    plt.xlim((0.0, max(xdata) + (.15 * max(xdata))))
    plt.ylim((0.0, max(ydata) + (.15 * max(ydata))))
    # Add labels and save
    ax1.set_title('Travian_Centralities')
    ax1.set_xlabel('closeness')
    ax1.set_ylabel('betweeness')
    plt.show()

    pos = nx.spring_layout(g_directed)
    betCent = nx.betweenness_centrality(g_directed, normalized=True, endpoints=True)
    node_color = [20000.0 * g_directed.degree(v) for v in g_directed]
    node_size = [v * 10000 for v in betCent.values()]
    plt.figure(figsize=(20, 20))
    nx.draw_networkx(g_directed, pos=pos, with_labels=True, node_size=node_size)
    plt.axis('off')
    plt.savefig('Visualize')
    plt.show()


def disti():
    # create some normal random noisy data
    ser = 50 * np.random.rand() * np.random.normal(10, 10, 100) + 20

    # plot normed histogram
    plt.hist(ser, normed=True)

    # find minimum and maximum of xticks, so we know
    # where we should compute theoretical distribution
    xt = plt.xticks()[0]
    xmin, xmax = min(xt), max(xt)
    lnspc = np.linspace(xmin, xmax, len(ser))

    # lets try the normal distribution first
    m, s = stats.norm.fit(ser)  # get mean and standard deviation
    pdf_g = stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
    plt.plot(lnspc, pdf_g, label="Norm")  # plot it

    # exactly same as above
    ag, bg, cg = stats.gamma.fit(ser)
    pdf_gamma = stats.gamma.pdf(lnspc, ag, bg, cg)
    plt.plot(lnspc, pdf_gamma, label="Gamma")

    # guess what :)
    ab, bb, cb, db = stats.beta.fit(ser)
    pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db)
    plt.plot(lnspc, pdf_beta, label="Beta")

    plt.show()

    in_degree_freq = degree_histogram_directed(g_directed, in_degree=True)
    out_degree_freq = degree_histogram_directed(g_directed, out_degree=True)
    degrees = range(len(in_degree_freq))
    plt.figure(figsize=(12, 8))
    plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'go-', label='in-degree')
    plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

import scipy.stats as ss
        # setting up the axes
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        # now plot
        alpha, loc, beta = 5, 100, 22
        data = ss.gamma.rvs(alpha, loc=loc, scale=beta, size=5000)
        myHist = ax.hist(data, 100, normed=True)
        rv = ss.gamma(alpha, loc, beta)
        x = np.linspace(0, 600)
        h = ax.plot(x, rv.pdf(x), lw=2)
        # show
        plt.show()


        m=3
        degree_freq = nx.degree_histogram(g_directed)
        degrees = range(len(degree_freq))
        plt.figure(figsize=(12, 8))
        plt.loglog(degrees[m:], degree_freq[m:], 'go-')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.show()
        # degrees = [g_directed.degree(n) for n in g_directed.nodes()]
        # plt.hist(degrees)
        # plt.margins(x=0)
        # plt.yscale('log', basey=10)
        # plt.show()

        in_degree_freq = degree_histogram_directed(g_directed, in_degree=True)
        out_degree_freq = degree_histogram_directed(g_directed, out_degree=True)
        degrees = range(len(in_degree_freq))
        plt.figure(figsize=(12, 8))
        plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'go-', label='in-degree')
        plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.show()

        degrees = g_directed.in_degree()
        degree_counts = Counter(degrees)
        x, y = zip(*degree_counts.items())

        plt.figure(1)

        # prep axes
        plt.xlabel('degree')
        #plt.xscale('log')
        plt.xlim(1, max(x))

        plt.ylabel('frequency')
        plt.yscale('log')
        plt.ylim(1, max(y))
        # do plot
        plt.scatter(x, y, marker='.')
        plt.show()

import matplotlib.pyplot as plt
import pandas as pd

x =[(10, 0.007923), (20, 0.012265), (30,	0.010759), (50, 0.008236),
 (70, 0.005583), (100, 0.006007), (150, 0.006188), (200, 0.012049), (250, 0.00811),
 (300,	0.007598), (350,0.00695),(400, 0.006749),(450,	0.007759),(500,	0.006047),(550, 0.011407), (600, 0.004201)]

x, y = zip(*x)
degrees = range(len(x))
plt.figure(figsize=(12, 8))
plt.loglog(x, y, 'go-', label='in-degree')
# plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
plt.title('Modularity distribution')
plt.xlabel('Number of Clusters')
plt.ylabel('Modularity')
plt.show()


#df = pd.read_csv('attacks-for-diagram.csv', engine='c', encoding='latin')

x=1
