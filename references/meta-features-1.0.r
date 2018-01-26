# ##########################################################################################
# for the installation of the required packages, use:
# install.packages(c('e1071','rrcov','mvpart','FNN','CORElearn', 'R.utils', 'infotheo'))
#
# usage:
# Rscript meta-features-1.0.R <dataset_filename> <meta-features_filename>
# ##########################################################################################

library(e1071)
library(rrcov)
library(mvpart)
library(FNN)
library(CORElearn)
library(R.utils)
library(infotheo)

n_col <- function(x) {
    dim(x)[2]
}

n_row <- function(x) {
    dim(x)[1]
}

get_class_indx <- function(data) {
    length(data)
}

get_col_name <- function(data, col) {
    names(data)[col]
}

get_att_names <- function(data) {
    names(data)[-get_class_indx(data)]
}

get_class_name <- function(data) {
    get_col_name(data, get_class_indx(data))
}

get_num_att <- function(data) {
    length(data) - 1
}

get_classes <- function(data) {
    levels(data[,get_class_indx(data)])
}

get_column_of_class <- function(data, column, class) {
    class_index <- get_class_indx(data)
    m <- data[class_index]==class
    data[,column][m]
}

get_att_data <- function(data) {
    class_index <- get_class_indx(data)
    data[-class_index]
}

get_class_data <- function(data) {
    data[get_class_indx(data)]
}

get_num_classes <- function(data) {
    length(get_classes(data))
}

get_num_numeric_att <- function(data) {
    att_data <- get_att_data(data)
    if(is.factor(att_data)) {
        n <- 0
    } else if(is.numeric(att_data)) {
        n <- 1
    } else {
        n <- 0
        for(att in 1:n_col(att_data)) {
            n <- n + is.numeric(att_data[,att])
        }
    }
    n
}

get_num_nominal_att <- function(data) {
    get_num_att(data) - get_num_numeric_att(data)
}

get_num_samples <- function(data) {
    n_row(data)
}

normalize <- function(data) {
    data_2 <- data.frame(data)
    num_att <- get_num_att(data_2)
    for(col in 1:num_att) {
        if(is.numeric(data[,col])) {
            min_v = min(data_2[col])
            max_v = max(data_2[col])
            d <- (max_v - min_v)
            if(is.na(d)) { # -- CGC: handling NAs as we do d == 0, will keep NA below
            	d <- 0
            }
            if(d == 0) {
                d <- 1.0
            }
            data_2[col] <- (data_2[col] - min_v) / d
        }
    }
    data_2
}

replace_nominal_column <- function(column, type=1) {
    column <- as.factor(column)
    symbols <- levels(column)
    result <- {}
    for (i in 1:(length(symbols)-type)) {
        result <- cbind(result, as.double(column == symbols[i]))
    }
    result
}

replace_nominal <- function(data, skip=numeric(0)) {
    result <- {}
    for(i in 1:n_col(data)) {
        column <- data[,i]
        if(is.numeric(column) || i %in% skip) {
            if(is.numeric(column)) {
                result <- cbind(result, column)
            } else {
                result <- cbind(result, data[i])
            }
        } else {
            result <- cbind(result, replace_nominal_column(column))
        }
    }
    data.frame(result)
}

replace_nominal_att <- function(data) {
    replace_nominal(data, skip=get_class_indx(data))
}

##########################################################################
##  Feature Selection for Meta-learning                                 ##
##  Alexandros Kalousis and Melanie Hilario                             ##
##  Advances in Knowledge Discovery and Data Mining                     ##
##  Lecture Notes in Computer Science, 2001, Volume 2035/2001, 222-233  ##
##  DOI: 10.1007/3-540-45357-1_26                                       ##
##########################################################################

get_symbol_stats <- function(data) {
    n <- numeric(0)
    for(i in 1:n_col(data)) {
        column <- data[,i]
        if(!is.numeric(column)) {
            n <- c(n, length(levels(as.factor(column))))
        }
    }
    result <- get_min_max_mean_sd(n, "symbols")
    result["symbols_sum"] <- sum(n)
    result
}

##########################################################################

get_single_normalized_entropies <- function(data) {
    m <- numeric(0)
    att_data <- get_att_data(data)
    for(i in 1:n_col(att_data)) {
        m[paste("att_entr", i, sep="_")] <- get_normalized_entropy(att_data[i])
    }
    m
}

get_single_mutual_information <- function(data) {
    m <- numeric(0)
    att_data <- get_att_data(data)
    labels <- get_class_data(data)
    for(i in 1:n_col(att_data)) {
        m[paste("att_mut_inf", i, sep="_")] <- get_mutual_information_column(att_data[i], labels)
    }
    m
}

##########################################################################
##  Meta-data: Characterization of Input Features for Meta-learning     ##
##  Ciro Castiello, Giovanna Castellano and Anna Maria Fanelli          ##
##  Modeling Decisions for Artificial Intelligence                      ##
##  Lecture Notes in Computer Science, 2005, Volume 3558/2005, 457-468  ##
##  DOI: 10.1007/11526018_45                                            ##
##########################################################################

get_entropy <- function(column, method="ML") {
    column <- column[,1]
    if(is.numeric(column)) {
        column <- infotheo::discretize(column, disc="equalwidth")
    }
    infotheo::entropy(column)
}

get_normalized_entropy <- function(column) {
    column <- column[,1]
    if(is.numeric(column)) {
        column <- infotheo::discretize(column, disc="equalwidth")
        n <- round(sqrt(dim(column)[1]))     # -- number of bins = square root of number of samples --
    }
    else {
        n <- length(levels(as.factor(column)))
        if(n==1) { # -- CGC: only one target class value / unlikely but happens in Bool space, NormEnt = Ent in this case
        	n <- 2
    	}
    }
    infotheo::entropy(column) / log(n)
}

get_avg <- function(data, f) {
    att_data <- get_att_data(data)
    avg <- 0.0
    for(i in 1:n_col(att_data)) {
        avg <- avg + f(att_data[i])
    }
    avg / n_col(att_data)
}

get_avg_with_labels <- function(data, f) {
    att_data <- get_att_data(data)
    labels <- get_class_data(data)
    avg <- 0.0
    for(i in 1:n_col(att_data)) {
        avg <- avg + f(att_data[i], labels)
    }
    avg / n_col(att_data)
}

get_attribute_entropy <- function(data) {
    get_avg(data, get_entropy)
}

get_normalized_attribute_entropy <- function(data) {
    get_avg(data, get_normalized_entropy)
}

get_class_entropy <- function(data) {
    infotheo::entropy(get_class_data(data))
}

get_normalized_class_entropy <- function(data) {
    get_normalized_entropy(get_class_data(data))
}

get_joint_entropy_column <- function(column, labels) {
    column <- column[,1]
    labels <- labels[,1]
    if(is.numeric(column)) {
        column <- infotheo::discretize(column, disc="equalwidth")
    }
    infotheo::entropy(paste(column, labels))
}

get_joint_entropy <- function(data) {
    get_avg_with_labels(data, get_joint_entropy_column)
}

get_mutual_information_column <- function(column, labels) {
    column <- column[,1]
    labels <- labels[,1]
    if(is.numeric(column)) {
        column <- infotheo::discretize(column, disc="equalwidth")
    }
    infotheo::mutinformation(column, labels)
}

get_mutual_information <- function(data) {
    get_avg_with_labels(data, get_mutual_information_column)
}

get_equivalent_number_of_attributes <- function(data) {
	cl_ent <- get_class_entropy(data)   # -- CGC: did not handle 0/0, Inf/NA/NaN
	mut_info <- get_mutual_information(data)
	if(cl_ent==0) {
		0
	} else if(mut_info==0) {
		1e10
	} else {
		cl_ent / mut_info
	}
#    get_class_entropy(data) / get_mutual_information(data)
}

#Sometimes Inf of course, so is the next one, although never NA

get_noise_signal_ratio <- function(data) {
	mut_info <- get_mutual_information(data)   # -- CGC: did not handle /0, Inf
	if(mut_info==0) {
		1e10
	} else {
		(get_attribute_entropy(data) - get_mutual_information(data)) / mut_info
	}
#    (get_attribute_entropy(data) - get_mutual_information(data)) / get_mutual_information(data)
}

##########################################################################

get_skewness <- function(data, type=1) {
    num_att <- get_num_att(data)
    classes <- get_classes(data)
    skew <- 0.0
    for(class in classes) {
        s <- 0.0
        n <- 0.0
        for(col in 1:num_att) {
            att_data_class <- get_column_of_class(data, col, class)
            if(is.numeric(att_data_class)) {  # -- skip nominal attributes --
                v <- e1071::skewness(att_data_class, type=type)
                if(!is.nan(v) && !is.na(v)) {  # -- NaN e.g. if the attribute has equal values for one class --
                    s <- s + abs(v)
                    n <- n + 1.0
                }
            }
        }
        if(n > 0.0) { # -- 0 e.g. if one class with only one smaple
            skew <- skew + (s / n)
        }
    }
    skew / (length(classes))
}

get_kurtotis <- function(data, type=1) {
    num_att <- get_num_att(data)
    classes <- get_classes(data)
    kurtosis <- 0.0
    for(class in classes) {
        s <- 0.0
        n <- 0.0
        for(col in 1:num_att) {
            att_data_class <- get_column_of_class(data, col, class)
            if(is.numeric(att_data_class)) {  # -- skip nominal attributes --
                v <- e1071::kurtosis(att_data_class, type=type)
                if(!is.nan(v) && !is.na(v)) {  # -- NaN e.g. if the attribute has equal values for one class --
                    s <- s + v
                    n <- n + 1.0
                }

            }
        }
        if(n > 0.0) { # -- 0 e.g. if one class with only one smaple
            kurtosis <- kurtosis + (s / n)
        }
    }
    (kurtosis / (length(classes))) + 3.0
}

get_cancors <- function(data) {
    att_data <- get_att_data(data)
    preprocess_att_data <- replace_nominal(att_data)
    labels <- get_class_data(data)
    preprocess_labels <- replace_nominal(labels)
 	if(get_num_classes(data)==1) { # -- CGC: only one target class value / unlikely but happens in Bool space
 		if(data[1,get_class_indx(data)]==0) {
 			center = 0.00000000001
 		}
 		else {
 			center = 1.00000000001
 		}
 		cancor(preprocess_att_data, preprocess_labels, TRUE, center)$cor
	}
	else {
    	cancor(preprocess_att_data, preprocess_labels)$cor
    }
}

get_cancor <- function(data, n) {
    m = numeric(0)
    c <- get_cancors(data)[1:n]
    for(i in 1:n) {
        m[paste("cancor",i, sep='_')] <- c[i]
    }
    m
}

get_fracts <- function(data) {
    cancors <- get_cancors(data)
    cancor2 = cancors * cancors
    cs <- cumsum(cancor2)   # -- CGC: avoiding 0/0 which leads to NA, return 0
    if(cs==0) {
    	0
    } else {
    	cs / sum(cancor2)
    }
    #cumsum(cancor2) / sum(cancor2)
}

get_fract <- function(data, n) {
    m = numeric(0)
    c <- get_fracts(data)[1:n]
    for(i in 1:n) {
        m[paste("fract",i, sep='_')] <- c[i]
    }
    m
}

get_abs_cor <- function(data) {
    num_att <- get_num_att(data)
    if(num_att > 1) { # -- correlation between attributes -> one attribute useless
        classes <- get_classes(data)
        sum = 0.0
        n = 0.0
        for(class in classes) {
            for(col1 in 1:num_att) {
                for(col2 in 1:num_att) {
                    if(col1 != col2) {
                        col1_data_class <- get_column_of_class(data, col1, class)
                        col2_data_class <- get_column_of_class(data, col2, class)
                        if(!is.numeric(col1_data_class)) {
                            col1_data_class <- replace_nominal_column(col1_data_class)
                        }
                        if(!is.numeric(col2_data_class)) {
                            col2_data_class <- replace_nominal_column(col2_data_class)
                        }
                        c <- tryCatch(cancor(col1_data_class, col2_data_class)$cor[1], error=function(err) NA)
                        if (!is.na(c)) {
                            sum <- sum + abs(c)
                            n <- n +1
                        }
                    }
                }
            }
        }
        sum / n
    } else {
        0.0
    }
}

get_naive_bayes <- function(data) {
    att_data <- get_att_data(data)
    labels <- get_class_data(data)[,1]
    if(length(levels(labels))==1) { # -- CGC: only one target class value / unlikely but happens in Bool space
    	levels(labels) <- c(0,1)
    }
    model <- naiveBayes(att_data, labels)
    predictions <- predict(model, att_data)
    sum(predictions == labels) / get_num_samples(data)
}

get_linear_discriminant <- function(data) {
    att_data <- get_att_data(data)
    preprocess_att_data <- replace_nominal(att_data)
    labels <- get_class_data(data)[,1]
    model <- tryCatch(LdaClassic(preprocess_att_data, labels), error=function(err) NULL)
	if(is.null(model)) {  # -- CGC: means some vectors are colinear / unlikely but happens in Bool space (did not change this, but catch in measure_landmark_p)
        NA
    } else {
        predictions <- predict(model, preprocess_att_data)@classification
        sum(predictions == labels) / get_num_samples(data)
    }
}

get_min_max_mean_sd <- function(x, name) {
    m <- numeric(0)
    m[paste(name, "min", sep="_")] = min(x)
    m[paste(name, "max", sep="_")] = max(x)
    m[paste(name, "mean", sep="_")] = mean(x)
    
    s <- sd(x)
    if(is.na(s)) {
        s <- 0
    }
    m[paste(name, "sd", sep="_")] <- s
    m
}

#######################################################################################
##  Decision tree based measures                                                     ##
#######################################################################################
##  Improved Dataset Characterisation for Meta-learning                              ##
##  Yonghong Peng, Peter A. Flach, Carlos Soares and Pavel Brazdil                   ##
##  Lecture Notes in Computer Science, 2002, Volume 2534/2002, 193-208               ##
##  DOI: 10.1007/3-540-36182-0_14                                                    ##
##  [all 15 features]                                                                ##
#######################################################################################
##  A higher-order approach to meta-learning                                         ##
##  H. Bensusan, C. Giraud-Carrier, and C. Kennedy.                                  ##
##  In Proceedings of the ECML'2000 Workshop on Meta- Learning:                      ##
##  Building Automatic Advice Strategies for Model Selection and Method Combination  ##
##  pages 109-117, 2000.                                                             ##
##  [only few features]                                                              ##
#######################################################################################

# Received from Matthias by email on Oct 12, 2012
tree.depth <- function (nodes) {
    depth <- floor(log(nodes, base = 2) + 1e-7)
    as.vector(depth - min(depth))
}

get_tree_properties <- function(data) {
    f <- as.formula(paste(get_class_name(data), "~ ."))
    tree_time <- System$currentTimeMillis();
    model <-mvpart::rpart(f, data = data, method = "class", control=rpart.control(maxcompete=0, maxsurrogate=0))
    tree_time <- System$currentTimeMillis() - tree_time
    leaves = rownames(model$frame[model$frame$var=="<leaf>",])
    n_leaves = length(leaves)
    n_nodes = sum(model$frame$var!="<leaf>")
    
    branch_lengths = numeric(0)
    for(leaf in leaves) {
        branch_lengths <- c(branch_lengths, length(path.rpart(model, leaf, print.it=FALSE)[[leaf]]))
    }
    
    att_used = numeric(0)
    for(att in get_att_names(data)) {
        att_used <- c(att_used, sum(model$frame$var==att))
    }
    
    depths <- tree.depth(as.numeric(row.names(model$frame)))
    level <- numeric(0)
    for(d in 1:max(depths)) {
        level <- c(level, sum(depths==d))
    }
    
    result <- numeric(0)
    result["nodes"] <- n_nodes
    result["leaves"] <- n_leaves
    result["nodes_per_attribute"] <- n_nodes / get_num_att(data)
    result["nodes_per_instance"] <- n_nodes / get_num_samples(data)
    result["leaf_corrobation"] <- mean(model$frame$n[model$frame$var=="<leaf>"]) / get_num_samples(data)
    result <- c(result, get_min_max_mean_sd(level, "level"))
    result <- c(result, get_min_max_mean_sd(branch_lengths, "branch"))
    result <- c(result, get_min_max_mean_sd(att_used, "attribute"))
    
    time <- numeric(0)
    time["tree_time"] <- tree_time
    list(result=result, time=time)
}

#######################################################################################

get_decision_stump <- function(data, col) {
    f <- as.formula(paste(get_class_name(data), "~", get_col_name(data, col)))
    model <-mvpart::rpart(f, data = data, method = "class", control=rpart.control(maxdepth=1))
    att_data <- get_att_data(data)
    labels <- get_class_data(data)[,1]
    predictions <- predict(model, att_data, type = "class")
    sum(predictions == labels) / get_num_samples(data)
}

get_decision_stumps <- function(data) {
    v <- numeric(0)
    g <- numeric(0)
    num_att <- get_num_att(data)
    time <- System$currentTimeMillis();
    for(col in 1:num_att) {
        v <- c(v, get_decision_stump(data, col))
        f <- as.formula(paste(get_class_name(data), "~", get_col_name(data, col)))
 		if(get_num_classes(data)==1) { # -- CGC: only one target class value / unlikely but happens in Bool space
 			g <- c(g, 0)
 		}
 		else {
        	g <- c(g, attrEval(f, data, estimator="GainRatio")[1])
        }
    }
    time <- System$currentTimeMillis() - time
    # -- attribute with minimal gain ration --
    #f <- as.formula(paste(get_class_name(data), "~ ."))
    #gainRatios <- attrEval(f, data, estimator="GainRatio")

    result <- get_min_max_mean_sd(v, "stump")
    result['stump_min_gain'] <- v[which.min(g)]
    result['stump_random'] <- v[sample(1:length(v), 1)]
    list(result=result, time=time)
}

get_knn <- function(data, k, cv=TRUE) {
    att_data <- get_att_data(data)
    labels <- get_class_data(data)
    if(cv) {
        predictions <- FNN::knn.cv(att_data, labels[,], k=k, prob=FALSE, algorithm="kd_tree")
    } else {
        predictions <- FNN::knn(att_data, att_data, labels[,], k=k, prob=FALSE, algorithm="kd_tree")
    }
    sum(as.vector(predictions) == labels) / get_num_samples(data)
}

get_knns <- function(data, max_k) {
    result <- numeric(0)
    time <- System$currentTimeMillis()
    for(k in 1:max_k) {
        result[paste("nn", k, sep="_")] = get_knn(data, k)
        gc()
    }
    time <- System$currentTimeMillis() - time
    if(length(result)==1) { # -- CGC: only one value produces NA - set sd to 0 for (meta)metalearning experiment
    	result["nn_sd"] <- 0
    }
    else {
    	result["nn_sd"] <- sd(result)
    }
    list(result=result, time=time)
}

get_class_stats <- function(data) {
    classes <- get_classes(data)
    class_data <- get_class_data(data)
    probs <- numeric(0)
    for(class in classes) {
        probs[class] <- sum(class_data == class) / n_row(data)
    }
    get_min_max_mean_sd(probs, "class_prob")
}

measure_landmark <- function(f, data, param) {
    measure_landmark_p(f, data, NULL)
}
    
measure_landmark_p <- function(f, data, param) {
    m = numeric(0)
    time <- System$currentTimeMillis()
    if(is.null(param)) {
        result <- f(data)
    } else {
        result <- f(data, param)
    }
    time <- System$currentTimeMillis() - time
    if(is.na(result)) {  # -- CGC: mostly/only applies to LDA(?), do not want NA for either result or time, change to 0 (poor performance) and 1e10 (bad time)
    	result <- 0
    	time <- 1e10
#        time <- NA
    }
    list(result=result, time=time)
}

compute_landmarking <- function(data, data_preprocessed=NULL) {
    if(is.null(data_preprocessed)) {
        data_numeric <- replace_nominal_att(data)
        data_preprocessed <- normalize(data_numeric)
    }
    naive_bayes <- measure_landmark(get_naive_bayes, data)
    lda <- measure_landmark(get_linear_discriminant, data_preprocessed)
    stumps <- get_decision_stumps(data)
    knns <- get_knns(data_preprocessed, 1)
    
    result <- numeric(0)
    result["naive_bayes"] <- naive_bayes$result
    result["lda"] <- lda$result
    result <- c(result, stumps$result)
    result <- c(result, knns$result)
    
    time <- numeric(0)
    time["naive_bayes_time"] <- naive_bayes$time
    time["lda_time"] <- lda$time
    time["stump_time"] <- stumps$time
    time["nn_time"] <- knns$time
    
    list(result=result, time=time)
}

compute_meta_features <- function(data, n) {
    combine(compute_grouped_meta_features(data))
}

combine <- function(x) {
    result <- numeric(0)
    for(name in names(x)) {
        result <- c(result, x[[name]])
    }
    result
}

test_mf <- function(data) {
    print(data)
}

compute_grouped_meta_features <- function(data) {
    total_time <- System$currentTimeMillis()
    data_numeric <- replace_nominal_att(data)
    data_preprocessed <- normalize(data_numeric)

    time <- numeric(0)

    simple <- numeric(0)
    simple_time <- System$currentTimeMillis()
 	if(get_num_classes(data)==1) { # -- CGC: only one target class value / unlikely but happens in Bool space
 		simple['classes'] <- 2
 	}
 	else {
    	simple['classes'] <- get_num_classes(data)
    }
    simple['attributes'] <- get_num_att(data)
    simple['numeric'] <- get_num_numeric_att(data)
    simple['nominal'] <- get_num_nominal_att(data)
    simple['samples'] <- get_num_samples(data)
    simple['dimensionality'] <- simple['attributes'] / simple['samples']
    simple['numeric_rate'] <- simple['numeric'] / simple['attributes']    ## -- Feature Selection for Meta-Learning
    simple['nominal_rate'] <- simple['nominal'] / simple['attributes']    ## -- Feature Selection for Meta-Learning
    simple <- c(simple, get_symbol_stats(data))
    simple <- c(simple, get_class_stats(data))
    simple_time <- System$currentTimeMillis() - simple_time

    statistical <- numeric(0)
    statistical_time <- System$currentTimeMillis()
    statistical['skewness'] <- get_skewness(data)
    statistical['skewness_prep'] <- get_skewness(data_preprocessed)
    statistical['kurtosis'] <- get_kurtotis(data, type=1)
    statistical['kurtosis_prep'] <- get_kurtotis(data_preprocessed, type=1)
    statistical['abs_cor'] <- get_abs_cor(data)
    statistical <- c(statistical, get_cancor(data, 1))
    statistical <- c(statistical, get_fract(data, 1))
    statistical_time <- System$currentTimeMillis() - statistical_time

    inftheo <- numeric(0)
    inftheo_time <- System$currentTimeMillis()
    inftheo['class_entropy'] <- get_class_entropy(data)
    inftheo['normalized_class_entropy'] <- get_normalized_class_entropy(data)
    inftheo['attribute_entropy'] <- get_attribute_entropy(data)
    inftheo['normalized_attribute_entropy'] <- get_normalized_attribute_entropy(data)
    inftheo['joint_entropy'] <- get_joint_entropy(data)
    inftheo['mutual_information'] <- get_mutual_information(data)
    inftheo['equivalent_attributes'] <- get_equivalent_number_of_attributes(data)
    inftheo['noise_signal_ratio'] <- get_noise_signal_ratio(data)
    
    inftheo_time <- System$currentTimeMillis() - inftheo_time

    modelbased <- get_tree_properties(data)
    time <- c(time, modelbased$time)
    modelbased <- modelbased$result

    landmarking <- compute_landmarking(data, data_preprocessed=data_preprocessed)
    time <- c(time, landmarking$time)
    landmarking <- landmarking$result

    total_time <- System$currentTimeMillis() - total_time
    time['simple_time'] <- simple_time
    time['statistical_time'] <- statistical_time
    time['inftheo_time'] <- inftheo_time
    time['total_time'] <- total_time

    list(simple=simple, statistical=statistical, inftheo=inftheo, modelbased=modelbased, landmarking=landmarking, time=time)
}

read <- function(filename) {
    data<-read.csv(filename, header=FALSE, sep=' ')
    m <- numeric(0)
    for(i in 1:get_num_att(data)) {
        m <- c(m, NA)
    }
    m <- c(m, "factor")
    read.csv(filename, header=FALSE, sep=' ', colClasses=m)
}

write_meta_features <- function(x, filename) {
    write.table(t(x), file=filename, row.names=FALSE, quote=FALSE)
}

args = base::commandArgs(TRUE)  # -- 'base::' needed because R.utils overwrites the function --

filename = args[1]
outfile = args[2]
n = 0
mf <- compute_meta_features(read(filename), n)
write_meta_features(mf, outfile)
