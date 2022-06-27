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

unfairness_threshold <- 0.05 # Can be set differently


main_plot_test <- function(dataset, metric, minX, maxX, minY, maxY) {
  
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
    
    labs(color='') + theme(legend.position = "none") +
    
    xlim(minX, maxX) + ylim(minY, maxY)
  
  pp
  ggsave(output_file, dpi=300, width=fig_width, height=fig_height, device=cairo_pdf) 
  return(pp)       
}

main_plot_train <- function(dataset, metric, minX, maxX, minY, maxY) {
  
  input_file              <- sprintf("./results_merged/%s_train_%s.csv", dataset, metric)
  output_file             <- sprintf("./graphs/%s_train_%s.pdf", dataset, metric)
  df                      <- read.csv(input_file, header=T)
  xlim <- 
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
    
    labs(color='') + theme(legend.position = "none") +
 
    xlim(minX, maxX) + ylim(minY, maxY)
  
  pp
  ggsave(output_file, dpi=300, width=fig_width, height=fig_height, device=cairo_pdf) 
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
  ggsave(output_file, dpi=300, width=fig_width, height=fig_height, device=cairo_pdf) 
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
  ggsave(legend_file, dpi=300, width=20, height=2, device=cairo_pdf) 
}

compute_axis_bounds <- function (dataset, metric){
  
  input_file              <- sprintf("./results_merged/%s_test_%s.csv", dataset, metric)
  output_file             <- sprintf("./graphs/%s_test_%s.pdf", dataset, metric)
  df                      <- read.csv(input_file, header=T)
  
  df <- df %>%
    filter(unf_test > 0, unf_test <= unfairness_threshold)
  
  pptest <- ggplot() + 
    
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
  
  input_file              <- sprintf("./results_merged/%s_train_%s.csv", dataset, metric)
  output_file             <- sprintf("./graphs/%s_train_%s.pdf", dataset, metric)
  df                      <- read.csv(input_file, header=T)
  xlim <- 
    df <- df %>%
    filter(unf_train > 0, unf_train <= unfairness_threshold)
  
  pptrain <- ggplot() + 
    
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
  
  xtest = layer_scales(pptest)$x$range$range
  ytest = layer_scales(pptest)$y$range$range
  xtrain = layer_scales(pptrain)$x$range$range
  ytrain = layer_scales(pptrain)$y$range$range
  xmin = min(min(xtest), min(xtrain))
  xmax = max(max(xtest), max(xtrain))
  ymin = min(min(ytest), min(ytrain))
  ymax = max(max(ytest), max(ytrain))
  return(c(xmin, xmax, ymin, ymax))
}

# loop to plot all the graphs
data_list <- c("adult", "compas", "default_credit", "marketing")
metric_list <- c("1", "2", "3", "4", "5", "6") #c("1", "2", "3", "4", "5") 

for (dataset in data_list) {
  for (metric in metric_list){
    bounds = compute_axis_bounds(dataset, metric)
    xmin = bounds[1]
    xmax = bounds[2]
    ymin = bounds[3]
    ymax = bounds[4]
    main_plot_train(dataset, metric, xmin, xmax, ymin, ymax)
    main_plot_test(dataset, metric, xmin, xmax, ymin, ymax)
    fairness_violation_plot(dataset, metric)
  }
}

# plot the legend bar 
plot_leggend("adult", "1")



