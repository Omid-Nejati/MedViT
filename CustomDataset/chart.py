import matplotlib.pyplot as plt

def BarChart(x, y, Title="Bar chart_data", X_label='x', Y_label='y', color='skyblue'):
    """
    Draw a bar chart using the given x and y data.

    Parameters:
    - x (list): Categories or labels on the x-axis
    - y (list): Corresponding values for each category
    - Title (str): Title of the bar chart
    - X_label (str): Label for the x-axis
    - Y_label (str): Label for the y-axis
    - color (str): Color of the bars (default: 'skyblue')
    """
    # Create the bar chart
    plt.bar(x, y, color=color)

    # Set the chart title and axis labels
    plt.title(Title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)

    # Display the chart
    plt.show()

def LineChart(x, y, Title="Line chart_data", X_label='x', Y_label='y', color='skyblue', name = "line.png"):
    """
    Creates and displays a line chart using matplotlib.
    
    Parameters:
    -----------
    y : array-like
        The y-axis data points to be plotted
    Title : str, optional
        The title of the chart (default: "Line chart_data")
    X_label : str, optional
        The label for the x-axis (default: 'x')
    Y_label : str, optional
        The label for the y-axis (default: 'y')
    color : str, optional
        The color of the line (default: 'skyblue')
        
    Returns:
    --------
    None
        Displays the line chart using matplotlib's show() function
    """
    plt.line( x, y, color=color)
    plt.title(Title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.savefig(name)
    plt.close()

def Line(y, Title="Line chart_data", X_label='x', Y_label='y', color='skyblue', name = ""):
    """
    Creates and displays a line chart using matplotlib.
    
    Parameters:
    -----------
    y : array-like
        The y-axis data points to be plotted
    Title : str, optional
        The title of the chart (default: "Line chart_data")
    X_label : str, optional
        The label for the x-axis (default: 'x')
    Y_label : str, optional
        The label for the y-axis (default: 'y')
    color : str, optional
        The color of the line (default: 'skyblue')
        
    Returns:
    --------
    None
        Displays the line chart using matplotlib's show() function
    """
    plt.plot( y, color=color)
    plt.title(Title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.savefig(name)
    plt.show()