from matplotlib import pyplot as plt


def plot_one_model_metric(results_dict, metric_to_plot):
    plt.xticks(list(range(len(results_dict))), [el['name'] for el in results_dict])
    plt.xlabel('model')
    plt.ylabel(metric_to_plot)
    
    plt.plot(list(range(len(results_dict))), [el[metric_to_plot] for el in results_dict])
    plt.show()
    
def plot_two_models_metric(results_models_dict, metric_to_plot):
    plt.xticks(list(range(len(results_models_dict[list(results_models_dict.keys())[0]]))),
               [el['name'] for el in results_models_dict[list(results_models_dict.keys())[0]]])
    plt.xlabel('model')
    plt.ylabel(metric_to_plot)
    
    for k in results_models_dict.keys():
        plt.plot(list(range(len(results_models_dict[k]))), [el[metric_to_plot] for el in results_models_dict[k]])

    plt.legend(list(results_models_dict.keys()), loc='upper left')
    plt.show()
    