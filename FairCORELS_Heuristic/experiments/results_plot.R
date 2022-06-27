library(ggplot2)
library(scales)
library(dplyr)
library(tidyselect)
library(ggpubr)
source("themes/theme.R")

point_size <- 25.5
line_size <- 3.5
point_size_vline <- 1.5
alpha_line <- 0.99
alpha_point <- 1.0
fig_width       <- 45
fig_height      <- 45
basesize <- 200

unfairness_threshold <- 0.5 # Can be set differently


main_plot_test <- function(dataset, metric) {

    input_file              <- sprintf("./results_merged/%s_test_%s.csv", dataset, metric)
    output_file             <- sprintf("./graphs/%s_test_%s.pdf", dataset, metric)
    df                      <- read.csv(input_file, header=T)

    df <- df %>%
            filter(unf_test > 0, unf_test <= unfairness_threshold)

    pp <- ggplot() + 

    geom_line(data=df, aes(x=error_test, y=unf_test, color=masks),  size=line_size, linetype = "dashed", alpha=alpha_line) + 

    geom_point(data=df, aes(x=error_test, y=unf_test, color=masks, shape=masks), size=point_size, alpha=alpha_point) + 


    theme_bw(base_size=basesize) +

    scale_color_manual(
        name="Strategy",
        values=c("#0e5357", "#cda815", "#a83546")
    )  +

    scale_shape_manual(
        name="Strategy",
        values=c(20, 20, 20)
    )  +
    
    labs(x = "Error test", y = "Unfairness test") + 
    
    labs(color='') + theme(legend.position = "none")

    pp
    ggsave(output_file, dpi=300, width=fig_width, height=fig_height) 
    return(pp)       
}

main_plot_train <- function(dataset, metric) {

    input_file              <- sprintf("./results_merged/%s_train_%s.csv", dataset, metric)
    output_file             <- sprintf("./graphs/%s_train_%s.pdf", dataset, metric)
    df                      <- read.csv(input_file, header=T)

    df <- df %>%
            filter(unf_train > 0, unf_train <= unfairness_threshold)

    pp <- ggplot() + 

    geom_line(data=df, aes(x=error_train, y=unf_train, color=masks),  size=line_size, linetype = "dashed", alpha=alpha_line) + 

    geom_point(data=df, aes(x=error_train, y=unf_train, color=masks, shape=masks), size=point_size, alpha=alpha_point) + 

    theme_bw(base_size=basesize) +

    scale_color_manual(
        name="Strategy",
        values=c("#0e5357", "#cda815", "#a83546")
    )  +

    scale_shape_manual(
        name="Strategy",
        values=c(20, 20, 20)
    )  +
    
    labs(x = "Error train", y = "Unfairness train") + 
    
    labs(color='') + theme(legend.position = "none")

    pp
    ggsave(output_file, dpi=300, width=fig_width, height=fig_height) 
    return(pp)       
}

fairness_violation_plot <- function(dataset, metric) {

    input_file              <- sprintf("./results_merged/fairness_violation_%s_%s.csv", dataset, metric)
    output_file             <- sprintf("./graphs/%s_fairness_violation_%s.pdf", dataset, metric)
    df                      <- read.csv(input_file, header=T)

    df <- df %>%
            filter(unf_train > 0, unf_train < unfairness_threshold)

    pp <- ggplot() + 

    geom_point(data=df, aes(x=unf_train, y=unf_test, color=masks, shape=masks), size=point_size, alpha=alpha_point) + 

    geom_abline(intercept = 0, slope = 1) +

    theme_bw(base_size=basesize) +

    scale_color_manual(
        name="Strategy",
        values=c("#0e5357", "#cda815", "#a83546")
    )  +

    scale_shape_manual(
        name="Strategy",
        values=c(20, 20, 20)
    )  +
    
    labs(x = "Unfairness train", y = "Unfairness test") + 
    
    labs(color='')  + theme(legend.position = "none")

    pp
    ggsave(output_file, dpi=300, width=fig_width, height=fig_height) 
    return(pp)       
}


plot_leggend <- function (dataset, metric){
    #update the parameter first to have a small legand
    point_size <- 7.5
    line_size <- 0.5
    point_size_vline <- 1.5
    alpha_line <- 0.99
    alpha_point <- 1.0
    fig_width       <- 15
    fig_height      <- 15
    basesize <- 70 

    
    input_file              <- sprintf("./results_merged/%s_test_%s.csv", dataset, metric)
    legend_file             <- sprintf("./graphs/legend.pdf")
    df                      <- read.csv(input_file, header=T)
    point_size <- point_size * 2
    pp <- ggplot() + 
    geom_line(data=df, aes(x=error_test, y=unf_test, color=masks),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
    geom_point(data=df, aes(x=error_test, y=unf_test, color=masks, shape=masks), size=point_size, alpha=alpha_point) + 
    theme_bw(base_size=basesize) +
    scale_color_manual(
        name="Strategy",
        values=c("#0e5357", "#cda815", "#a83546")
    )  +

    scale_shape_manual(
        name="Strategy",
        values=c(20, 20, 20)
    )  +
    labs(color='') + theme(legend.direction = "horizontal", legend.position = "top", legend.box = "horizontal")
    # Extract the legend. Returns a gtable
    leg <- get_legend(pp)
    # Convert to a ggplot and print
    as_ggplot(leg)
    ggsave(legend_file, dpi=300, width=20, height=2) 
}


# loop to plot all the graphs
data_list <- c("adult", "compas", "default_credit", "marketing")
metric_list <- c("1", "2", "3", "4", "5", "6") #c("1", "2", "3", "4", "5") 


for (dataset in data_list) {
    for (metric in metric_list){
        main_plot_train(dataset, metric)
        main_plot_test(dataset, metric)
        fairness_violation_plot(dataset, metric)
    }
}

# plot the legend bar 
plot_leggend("adult", "1")



