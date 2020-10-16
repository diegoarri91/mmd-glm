import matplotlib.pyplot as plt


def plot_layout(f):
    r1, r2, r3 = 1, 2, 2
    c1, c2, c3 = 3, 2, 1
    nrows = 3 * r2
    ncols = c1 + c2
    fig = plt.figure(figsize=(1 * f, 1 * f))
    ax00 = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=r1, colspan=c1)
    ax01 = plt.subplot2grid((nrows, ncols), (0, c1), rowspan=r2, colspan=c2)
    ax10 = plt.subplot2grid((nrows, ncols), (r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    ax20 = plt.subplot2grid((nrows, ncols), (2 * r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    ax21 = plt.subplot2grid((nrows, ncols), (2 * r1, c1), rowspan=r2, colspan=c2)
    ax30 = plt.subplot2grid((nrows, ncols), (3 * r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    ax40 = plt.subplot2grid((nrows, ncols), (4 * r1, 0), rowspan=r2, colspan=c2)
    ax41 = plt.subplot2grid((nrows, ncols), (4 * r1, c2), rowspan=r3, colspan=c3)
    ax42 = plt.subplot2grid((nrows, ncols), (4 * r1, c2 + 1), rowspan=r3, colspan=c3)
    ax43 = plt.subplot2grid((nrows, ncols), (4 * r1 + 1, c2 + 2), rowspan=1, colspan=c3)
    return fig, (ax00, ax01, ax10, ax20, ax21, ax30, ax40, ax41, ax42, ax43)

def plot_layout2(f):
    r1, r2, r3 = 1, 2, 2
    c1, c2, c3 = 3, 2, 1
    nrows = 3 * r2
    ncols = c1 + 2 * c2
    fig = plt.figure(figsize=(1 * f, 1 * f))
    ax00 = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=r1, colspan=c1)
    ax01 = plt.subplot2grid((nrows, ncols), (0, c1), rowspan=r2, colspan=c2)
    ax10 = plt.subplot2grid((nrows, ncols), (r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    ax20 = plt.subplot2grid((nrows, ncols), (2 * r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    ax21 = plt.subplot2grid((nrows, ncols), (2 * r1, c1), rowspan=r2, colspan=c2)
    ax30 = plt.subplot2grid((nrows, ncols), (3 * r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    ax40 = plt.subplot2grid((nrows, ncols), (4 * r1, 0), rowspan=r2, colspan=c2)
    ax41 = plt.subplot2grid((nrows, ncols), (4 * r1, c2), rowspan=r3, colspan=c3)
    ax42 = plt.subplot2grid((nrows, ncols), (4 * r1, c2 + 1), rowspan=r3, colspan=c3)
    ax43 = plt.subplot2grid((nrows, ncols), (4 * r1, c2 + 2), rowspan=r3, colspan=c3)
    return fig, (ax00, ax01, ax10, ax20, ax21, ax30, ax40, ax41, ax42, ax43)

def plot_layout3(f):
    r1, r2, r3 = 1, 2, 1
    c1, c2, c3 = 3, 2, 1
    nrows = 2 * r2
    ncols = c1 + 2 * c2
    fig = plt.figure(figsize=(2 * f, 1 * f))
    ax00 = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=r1, colspan=c1)
    ax01 = plt.subplot2grid((nrows, ncols), (0, c1), rowspan=r3, colspan=c3)
    ax01b = plt.subplot2grid((nrows, ncols), (r1, c1), rowspan=r3, colspan=c3)
    ax01c = plt.subplot2grid((nrows, ncols), (0, c1 + c3), rowspan=1, colspan=c3)
    ax02 = plt.subplot2grid((nrows, ncols), (0, c1 + 2 * c3), rowspan=r2, colspan=c2)
#     ax02 = plt.subplot2grid((nrows, ncols), (0, c1 +  2 * c3), rowspan=r2, colspan=c2)
    ax10 = plt.subplot2grid((nrows, ncols), (r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    ax20 = plt.subplot2grid((nrows, ncols), (2 * r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    ax21 = plt.subplot2grid((nrows, ncols), (2 * r1, c1), rowspan=r2, colspan=c2)
    ax22 = plt.subplot2grid((nrows, ncols), (2 * r1, c1 + c2), rowspan=r2, colspan=c2)
    ax30 = plt.subplot2grid((nrows, ncols), (3 * r1, 0), rowspan=r1, colspan=c1, sharex=ax00)
    return fig, (ax00, ax01, ax01b, ax01c, ax02, ax10, ax20, ax21, ax22, ax30)



def plot_layout_rebuttal(f):
    c1, c2 = 2, 1
    nrows = 1
    ncols = 6
    fig = plt.figure(figsize=(3 * f, 1 * f))
    axeta = plt.subplot2grid((nrows, ncols), (0, 0), colspan=c1)
    axac = plt.subplot2grid((nrows, ncols), (0, c1), colspan=c1)
    axmmd = plt.subplot2grid((nrows, ncols), (0, 2 * c1), colspan=c2)
    axll = plt.subplot2grid((nrows, ncols), (0, 2 * c1 + c2), colspan=c2, sharex=axmmd)
    return fig, (axeta, axac, axmmd, axll)