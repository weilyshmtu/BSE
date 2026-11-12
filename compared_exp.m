clear
clc
curr_fold = pwd();
addpath("algs\")
addpath("metric\")
addpath("funs\")
addpath("compared algs\")
addpath("datasets\")
datasets = ["Zoo","Yale_32x32","JAFFE_64x64","ORL_32x32","Umist_575_32x32","FERET_32x32","COIL20_32x32","AR_32x32","YALEB_32x32",...
 "COIL100_32x32","MNIST_10k_Train_28x28","pendigits"];
numdatas = length(datasets);
algorithms = ["BSE_MM_revised","SSC", "LRR", "AGCSC", "IDR", "ISC", "ERCAN", "SGL","DSC", "SDSGC", "SGR","MLSC", "TGL"];
numalgs = length(algorithms);
rng(42)

% --------------------------
% Create results folder if it doesn't exist
% --------------------------
device = "gpu";

normtype = 1;
total_results_dir = fullfile(curr_fold, "TotalResults" + "_" +"gpu" + "_" + num2str(normtype));
if ~exist(total_results_dir, 'dir')
    mkdir(total_results_dir);
end
rn = 10;
for d_ind =1:numdatas
    data = datasets(d_ind);
    
    % load data
    load(data);
    Y = gnd;
    X = fea';

    % normalization
    % unit l2 normalization
    X = normalize_data(X, normtype);
    
    for alg_ind = 1
        alg = algorithms(alg_ind);
        
        % options
        opts = [];
        [d, n] = size(X);
        opts.n = n;
        opts.d = d;
        opts.n_clusters = length(unique(Y));
        
        switch alg


            %% BSE: MM
            case 'BSE_MM_revised'
                %%

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end
                
                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is BSE!')
                
                %parameter
                k_set     = [3, 5, 8, 10, 15, 20, 25, 30];
                gamma_set = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1];
                opts.Dx = L2_distance_1(X, X);
                opts.X = X;
                opts.XtX = X'*X;
                opts.max_iter = 3000;
                opts.tol = 1e-4;
                opts.labels = Y;
                bse_mm_time = [];
                k_init = 10;
                opts.S_init = initialize_S(X, opts.Dx, k_init);
                opts.C_init = (opts.XtX +0.01*eye(n))\opts.XtX;
                % opts.C_init = opts.S_init;
                for a_index = 1 :length(gamma_set)
                    for k_index =  1 : length(k_set)
                        
                        opts.k = k_set(k_index);
                        alpha = gamma_set(a_index)*opts.k^3

                        if device == "cpu"
                           [C, S, info] = bse_mm_cpu(X, alpha, opts);
                        else
                           [C, S, info] = bse_mm_gpu(X, alpha, opts);

                        end
                        time = info.time
                        bse_mm_time = [bse_mm_time  time];
                        res = [info.resS info.resC]';
 
                        subplot(1,3,1)
                        plot(1:size(res,2), res(1,:));
                        subplot(1,3,2)
                        plot(1:size(res,2), res(2,:));
                        subplot(1,3,3)
                        obj_hist = info.obj;
                        plot(1:length(obj_hist),obj_hist)
                        drawnow


                        resultC1 = [];
                        resultS1 = [];
                        resultSC1 = [];

                        W_C1 = 0.5*(C' + C);
                        for t=1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end

                        W_S1 = 0.5*(S' + S);
                        for t=1:rn
                            predY = SpectralClustering(W_S1, opts.n_clusters);
                            result21 = Clustering8Measure(Y, predY);
                            resultS1 = [resultS1; result21];
                        end

                       
                        W_SC1 = 0.5*(W_C1+W_S1);
                        for t=1:rn
                            predY = SpectralClustering(W_SC1, opts.n_clusters);
                            result3 = Clustering8Measure(Y, predY);
                            resultSC1 = [resultSC1; result3];
                        end

                        clear W_C1 W_S1 W_SC1
                         
                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, k_index} = mresultC1;
                        resultsC1_std{a_index, k_index} = std(resultC1);

                        mresultS1 = mean(resultS1);
                        resultsS1_mean{a_index, k_index} = mresultS1;
                        resultsS1_std{a_index, k_index} = std(resultS1);


                        mresultSC1 = mean(resultSC1);
                        resultsSC1_mean{a_index, k_index} = mresultSC1;
                        resultsSC1_std{a_index, k_index} = std(resultSC1);

                        acc_C1(a_index, k_index) = mresultC1(1)
                        acc_S1(a_index, k_index) = mresultS1(1)
                        acc_SC1(a_index, k_index) = mresultSC1(1)
     
 
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(k_index);
                        
                        save(fullfile(alg_data_results_dir, filename), "C", "S","resultC1", "resultS1", "resultSC1")

                        clear C S res time predY result11 result21 result12 result22 result3 result4  result5 resultC1 resultS1 mresultC1 ...
                            mresultS1 resultC2 resultS2 mresultC2 mresultS2 mresultSC1 mresultSC2  mresultSC3 mresultSC4
                        pause(0.5);
                    end
                end
                clc
                best_results = [max(max(acc_C1))  max(max(acc_S1))  max(max(acc_SC1)) ]
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "resultsS1_mean", "resultsS1_std","resultsSC1_mean",...
                    "resultsSC1_std", "bse_mm_time", '-v7.3');
                clear acc_C1 acc_S1 acc_SC1 acc_SC2 acc_SC3 acc_SC4 resultsC1_mean resultsS1_mean resultsS1_std resultsSC1_mean resultsSC1_std resultsSC2_mean resultsSC2_std ...
                    acc_C2 acc_S2 resultsC2_mean resultsS2_mean resultsS2_std  ...
                    resultsSC3_mean resultsSC3_std resultsSC4_mean resultsSC4_std alg_results_dir alg_data_results_dir filename bse_mm_time;
    
         
            % ====================================================================
            %% SSC: Sparse subspace clustering
            case 'SSC'
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end
                
                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is SSC!')
                % parameters 
                lambda_set = [1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100];
                opts.XtX = X'*X;
                
                ssc_time = [];
                for a_index = 1:length(lambda_set)
                    lambda = lambda_set(a_index);
                    if device == "cpu"
                        [C, time] = ssc_cpu(X, lambda, opts);
                    else
                        [C, time] = ssc_gpu(X, lambda, opts);
                    end
                    ssc_time = [ssc_time time];
                    
                    % Sparse & Isolation
                    W_C1 = abs(C') + abs(C);
                    
                    resultC1 = [];
                    for t = 1:rn
                        predY = SpectralClustering(W_C1, opts.n_clusters);
                        result11 = Clustering8Measure(Y, predY);
                        resultC1 = [resultC1; result11];
                    end
                    clear W_C1;
                    
                    mresultC1 = mean(resultC1);
                    resultsC1_mean{a_index} = mresultC1;
                    resultsC1_std{a_index} = std(resultC1);
                    acc_C1(a_index) = mresultC1(1);
                    
                    alg_data_results_dir = fullfile(alg_results_dir, data);
                    if ~exist(alg_data_results_dir, 'dir')
                        mkdir(alg_data_results_dir);
                    end
                    filename = num2str(a_index);
                    save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "time", '-v7.3');
                
                    % Garbage Collection
                    clear C time predY result11 resultC1 mresultC1;
                    pause(0.5);
                end
                ssc_ave_time = mean(ssc_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "ssc_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename ssc_time;
                

            % ====================================================================
            %% LRR: Robust Recovery of Subspace Structures
            case 'LRR'
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is LRR!')

                % parameters 
                lambda_set = [1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100];
                opts.XtX = X'*X;

                lrr_time = [];
                for a_index = 1:length(lambda_set)
                    lambda = lambda_set(a_index);
                    if device == "cpu"
                        [C, E, time] = lrr_cpu(X, lambda, opts);
                    else
                        [C, E, time] = lrr_gpu(X, lambda, opts);
                    end
                    lrr_time = [lrr_time time];
                    
                    % Sparse & Isolation
                    W_C1 = abs(C') + abs(C);
                    
                    resultC1 = [];
                    for t = 1:rn
                        predY = SpectralClustering(W_C1, opts.n_clusters);
                        result11 = Clustering8Measure(Y, predY);
                        resultC1 = [resultC1; result11];
                    end
                    clear W_C1;
                    
                    mresultC1 = mean(resultC1);
                    resultsC1_mean{a_index} = mresultC1;
                    resultsC1_std{a_index} = std(resultC1);
                    acc_C1(a_index) = mresultC1(1)

                    alg_data_results_dir = fullfile(alg_results_dir, data);
                    if ~exist(alg_data_results_dir, 'dir')
                        mkdir(alg_data_results_dir);
                    end
                    filename = num2str(a_index);
                    save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "E","time", '-v7.3');
                
                    % Garbage Collection
                    clear C E time predY result11 resultC1 mresultC1;
                    pause(0.5);
                end
                lrr_ave_time = mean(lrr_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "lrr_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename lrr_time;
            

            % ====================================================================
            %% AGCSC: Adaptive Graph Convolutional Subspace Clustering
            case "AGCSC"
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is AGCSC!')
                % parameters 
                alpha_set = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5];
                beta_set = alpha_set;
               
                opts.e = ones(n,1);
                opts.eet = opts.e*opts.e';
                opts.max_iter = 1e4;
                opts.XtX = X'*X;
                X = X'; % transpose the data

                agcsc_time = [];
                for a_index = 1:length(alpha_set)
                    for b_index = 1:length(beta_set)
                        alpha = alpha_set(a_index);
                        beta = beta_set(b_index);

                        if device == "cpu"
                            [C, F, time] = agcsc_cpu(X, alpha, beta, opts);
                        else
                            [C, F, time] = agcsc_gpu(X, alpha, beta, opts);
                        end
                        agcsc_time = [agcsc_time time];
                        
                        % Sparse & Isolation
                        W_C1 = abs(C') + abs(C);
                        
                        resultC1 = [];
                        for t = 1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end
                        clear W_C1;
                        
                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, b_index} = mresultC1;
                        resultsC1_std{a_index, b_index} = std(resultC1);
                        acc_C1(a_index, b_index) = mresultC1(1)

                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(b_index);
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "F","time", '-v7.3');
                        
                        % Garbage Collection
                        wait(gpuDevice);
                        clear C F time predY result11 resultC1 mresultC1;
                        pause(0.5);
                    end
                end
                X = X'; % retranspose the data
                agcsc_ave_time = mean(agcsc_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "agcsc_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename agcsc_time;

            % ====================================================================
            %% IDR: Learning idempotent representation
            case "IDR"
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is IDR!')
                % parameters 
                gamma_set = [1e-3, 1e-2, 1e-1, 1, 10, 100];
                lambda_set = gamma_set;

                opts.e = ones(n,1);
                opts.eet = opts.e*opts.e';
                opts.max_iter = 500;
                opts.XtX = X'*X;

                idr_time = [];
                for g_index = 1:length(gamma_set)
                    for l_index = 1:length(lambda_set)
                        gamma = gamma_set(g_index);
                        lambda = lambda_set(l_index);

                        if device == "cpu"
                            [C, S, E, time] = idr_cpu(X,opts.n_clusters,gamma,lambda, opts);
                        else
                            [C, S, E, time] = idr_gpu(X,opts.n_clusters,gamma,lambda, opts);
                        end
                        idr_time = [idr_time time];
                        
                        % Sparse & Isolation
                        % C = sparse(C);
                        % S = sparse(S);
                        W_C1 = abs(C') + abs(C);
                        
                        resultC1 = [];
                        for t = 1:rn
                            predY = SpectralClustering(W_C1,opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end
                        clear W_C1;
                        
                        mresultC1 = mean(resultC1);
                        resultsC1_mean{g_index, l_index} = mresultC1;
                        resultsC1_std{g_index, l_index} = std(resultC1);
                        acc_C1(g_index, l_index) = mresultC1(1)

                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(g_index) + "x" + num2str(l_index);
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "S", "E", "time", '-v7.3');
                        
                        % Garbage Collection
                        clear C S E time predY result11 resultC1 mresultC1;
                        pause(0.5);
                    end
                end
                idr_ave_time = mean(idr_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "idr_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename idr_time;
          
            % ====================================================================
            %% ERCAN: Entropy Regularization
            case "ERCAN"
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is ERCAN!')
                % parameters 
                k_set = 2:4:26;
                eta_set = [1,10,100,1e3,1e4,1e5,1e6];
                opts.Dx = L2_distance_1(X, X);

                ercan_time = [];
                for a_index = 1:length(k_set)
                    for b_index = 1:length(eta_set)
                        k = k_set(a_index);
                        eta = eta_set(b_index);
                        
                        if device =="cpu"
                            [C, F, time] = ercan_cpu(X, k, eta, opts);
                        else
                            [C, F, time] = ercan_gpu(X, k, eta, opts);
                        end
                        ercan_time = [ercan_time, time];
                        
                        % Sparse & Isolation
                        % C = sparse(C);
                        W_C1 = abs(C') + abs(C);
                        
                        resultC1 = [];
                        for t = 1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end
                        clear W_C1;
                        
                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, b_index} = mresultC1;
                        resultsC1_std{a_index, b_index} = std(resultC1);
                        acc_C1(a_index, b_index) = mresultC1(1)

                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(b_index);
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "F", "time", '-v7.3');
                        
                        % Garbage Collection
                        clear C F time predY result11 resultC1 mresultC1;
                        pause(0.5);
                    end
                end
                ercan_ave_time = mean(ercan_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std","ercan_ave_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename ercan_time;

            % ====================================================================
            %% SGR: Clustering with Self-Learned Graph Regression
            case "SGR"
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is SGR!')
                % parameters 
                alpha_set = [5e-5, 1e-4, 5e-3, 1e-3, 5e-3, 1e-2];
                beta_set = alpha_set;
                
                opts.Dx = L2_distance_1(X, X);
                opts.k = 20;
                
                % Clear structs before initialization
                if isfield(opts, 'S_init'), opts = rmfield(opts, 'S_init'); end
                if isfield(opts, 'F_init'), opts = rmfield(opts, 'F_init'); end
                
                opts.S_init = initialize_S(X, opts.Dx, opts.k);
                opts.F_init = initialize_F(opts.S_init,opts.n_clusters);
                opts.XtX = X'*X;

                sgr_time = [];
                 for a_index = 1:length(alpha_set)
                    for b_index = 1:length(beta_set)
                        alpha = alpha_set(a_index);
                        beta = beta_set(b_index);

                        if device == "cpu"
                            [S, C, F, time, obj_val, acc_array1, acc_array2] = sgr_cpu(opts.XtX, alpha, beta, opts);
                        else
                            [S, C, F, time, obj_val, acc_array1, acc_array2] = sgr_gpu(opts.XtX, alpha, beta, opts);
                        end
                        sgr_time = [sgr_time time];
                        
                        % Sparse & Isolation (Clustering on S per original code)
                        W_S1 = abs(S') + abs(S);
                        
                        resultC1 = [];
                        for t = 1:rn
                            predY = SpectralClustering(W_S1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end
                        clear W_S1;
                        
                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, b_index} = mresultC1;
                        resultsC1_std{a_index, b_index} = std(resultC1);
                        acc_C1(a_index, b_index) = mresultC1(1)

                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(b_index);
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "S", "F", "time", '-v7.3');
                        
                        % Garbage Collection
                        clear C S F obj_val acc_array1 acc_array2 time predY result11 resultC1 mresultC1;
                        pause(0.5);
                    end
                end
                sgr_ave_time = mean(sgr_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std","sgr_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename sgr_time opts;

            % ====================================================================
            %% SGL: Effective Clustering via Structured Graph Learning
            case 'SGL'
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is SGL!')

                % parameters 
                lambda_set = [1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100];
                opts.Dx = L2_distance_1(X, X);

                sgl_time = [];
                for a_index = 1:length(lambda_set) 
                    lambda = lambda_set(a_index);
                    if device == "cpu"
                        [C, time] = sgl_cpu(opts.Dx, opts.n_clusters, lambda);
                    else
                        [C, time] = sgl_gpu(opts.Dx, opts.n_clusters, lambda);
                    end
                    sgl_time =[sgl_time time];
                    
                    % Sparse & Isolation
                    W_C1 = abs(C') + abs(C);
                    
                    resultC1 = [];
                    for t = 1:rn
                        predY = SpectralClustering(W_C1, opts.n_clusters);
                        result11 = Clustering8Measure(Y, predY);
                        resultC1 = [resultC1; result11];
                    end
                    clear W_C1;
                    
                    mresultC1 = mean(resultC1);
                    resultsC1_mean{a_index} = mresultC1;
                    resultsC1_std{a_index} = std(resultC1);
                    acc_C1(a_index) = mresultC1(1)
                    
                    alg_data_results_dir = fullfile(alg_results_dir, data);
                    if ~exist(alg_data_results_dir, 'dir')
                        mkdir(alg_data_results_dir);
                    end
                    filename = num2str(a_index);
                    save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "time", '-v7.3');
                
                    % Garbage Collection
                    clear C time predY result11 resultC1 mresultC1;
                    pause(0.5);
                end
                sgl_ave_time = mean(sgl_time);

                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "sgl_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename sgl_time;

            % ====================================================================
            %% TGL: Tensorized Graph Learning Framework for Clustering
            case 'TGL'
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is TGL!')
                % parameters 
                lambda1_set = [1e-3, 1e-2, 1e-1, 1, 10, 100];
                lambda2_set = [1e-3, 1e-2, 1e-1, 1, 10, 100];

                opts.Dx = L2_distance_1(X, X);
                opts.XtX = X'*X;
                tgl_time = [];
                for a_index = 1:length(lambda1_set)
                    for b_index = 1:length(lambda2_set)
                        lambda1 = lambda1_set(a_index);
                        lambda2 = lambda2_set(b_index);

                        if device == "cpu"
                            [C, time] = tgl_cpu(lambda1, lambda2, opts);
                        else
                            [C, time] = tgl_gpu(lambda1, lambda2, opts);
                        end
                        tgl_time = [tgl_time time];
                        
                        % Sparse & Isolation
                        % C = sparse(C);
                        W_C1 = abs(C') + abs(C);
                        
                        resultC1 = [];
                        for t = 1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end
                        clear W_C1;
                        
                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, b_index} = mresultC1;
                        resultsC1_std{a_index, b_index} = std(resultC1);
                        acc_C1(a_index, b_index) = mresultC1(1)
                        
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(b_index);
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "time", '-v7.3');
                        
                        % Garbage Collection
                        clear C time predY result11 resultC1 mresultC1;
                        pause(0.5);
                    end
                end
                tgl_ave_time = mean(tgl_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "tgl_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename tgl_time;

            % ====================================================================
            %% SDSGC: Structured Doubly Stochastic Graph-Based Clustering
            case 'SDSGC'
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is SDSGC!')
                % parameters 
                k_set = [5,  8, 10, 12, 15, 18, 20];
                opts.Dx = L2_distance_1(X, X);
                sdsgc_time = [];

                for a_index = 1:length(k_set)
                    k = k_set(a_index);
                    
                    if isfield(opts, 'S_init')
                        opts = rmfield(opts, 'S_init');
                    end
                    opts.S_init = initialize_S(X, opts.Dx, k);

                    if device == "cpu"
                        [C, F, time] = sdsgc_cpu(X, k, opts);
                    else
                        [C, F, time] = sdsgc_gpu(X, k, opts);
                    end
                    sdsgc_time = [sdsgc_time time];
                    
                    % Sparse & Isolation
                    % C = sparse(C);
                    W_C1 = abs(C') + abs(C);
                    
                    % predY = kmeans(F,opts.n_clusters);
                    resultC1 = [];
                    for t = 1:rn
                        predY = SpectralClustering(W_C1, opts.n_clusters);
                        result11 = Clustering8Measure(Y, predY);
                        resultC1 = [resultC1; result11];
                    end
                    clear W_C1;
                    
                    mresultC1 = mean(resultC1);
                    resultsC1_mean{a_index} = mresultC1;
                    resultsC1_std{a_index} = std(resultC1);
                    acc_C1(a_index) = mresultC1(1)
                    
                    alg_data_results_dir = fullfile(alg_results_dir, data);
                    if ~exist(alg_data_results_dir, 'dir')
                        mkdir(alg_data_results_dir);
                    end
                    filename = num2str(a_index);
                    save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "F", "time", '-v7.3');
                
                    % Garbage Collection
                    clear C F time predY result11 resultC1 mresultC1;
                    pause(0.5);
                end
                sdsgc_ave_time = mean(sdsgc_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std","sdsgc_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename sdsgc_time;

            % ====================================================================
            %% DSC: A Novel and Effective Method to Directly Solve Spectral Clustering
            case 'DSC'
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is DSC!')
                % parameters
                k_set = [5, 8, 10, 12, 15, 18, 20];
                opts.Dx = L2_distance_1(X, X);
                dsc_time = [];
                for a_index = 1:length(k_set)
                    k = k_set(a_index);
                    
                    if isfield(opts, 'S_init')
                        opts = rmfield(opts, 'S_init');
                    end
                    opts.S_init = initialize_S(X, opts.Dx, k);

                    if device == "cpu"
                        [F, C, time] = dsc_cpu(opts.S_init, k, opts);
                    else
                        [F, C, time] = dsc_gpu(opts.S_init, k, opts);
                    end
                    dsc_time = [dsc_time time];
                    
                    % Sparse (no W_C needed for kmeans based on original logic)
                    % C = sparse(C);
                    
                    % predY = SpectralClustering((abs(C')+ abs(C)),opts.n_clusters);
                    resultC1 = [];
                    for t = 1:rn
                        predY = kmeans(F, opts.n_clusters);
                        result11 = Clustering8Measure(Y, predY);
                        resultC1 = [resultC1; result11];
                    end
                    
                    mresultC1 = mean(resultC1);
                    resultsC1_mean{a_index} = mresultC1;
                    resultsC1_std{a_index} = std(resultC1);
                    acc_C1(a_index) = mresultC1(1)
                    
                    alg_data_results_dir = fullfile(alg_results_dir, data);
                    if ~exist(alg_data_results_dir, 'dir')
                        mkdir(alg_data_results_dir);
                    end
                    filename = num2str(a_index);
                    save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "time", '-v7.3');
                
                    % Garbage Collection
                    clear C F time predY result11 resultC1 mresultC1;
                    pause(0.5);
                end
                dsc_ave_time = mean(dsc_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "dsc_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename dsc_time;

            % ====================================================================
            %% MLSC: Metric Learning-Based Subspace Clustering, TNNLS 2025
            case 'MLSC'
                %% 

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end

                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is MLSC!')
                % parameters
                alpha_set = [1e-5, 1e-3, 1e-1, 1, 1e1, 1e3, 1e5];
                beta_set = alpha_set;
                
                mlsc_time = [];
                for a_index = 1 : length(alpha_set)
                    for b_index = 1 : length(beta_set)
                        alpha = alpha_set(a_index);
                        beta = beta_set(b_index);

                        if device == "cpu"
                            [C, G, time] = mlsc_cpu(X, alpha, beta, opts);
                        else
                            [C, G, time] = mlsc_gpu(X, alpha, beta, opts);
                        end
                        mlsc_time = [mlsc_time time];
                        
                        % Sparse & Isolation
                        % C = sparse(C);
                        W_C1 = abs(C') + abs(C);
                        
                        resultC1 = [];
                        for t = 1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end
                        clear W_C1;
                        
                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, b_index} = mresultC1;
                        resultsC1_std{a_index, b_index} = std(resultC1);
                        acc_C1(a_index, b_index) = mresultC1(1)
                        
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(b_index);
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "G", "time", '-v7.3');
                        
                        % Garbage Collection
                        clear C G time predY result11 resultC1 mresultC1;
                        pause(0.5);
                    end
                end
                mlsc_ave_time = mean(mlsc_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "mlsc_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename mlsc_time;

            % ====================================================================
            %% ISC: Interpretable subspace clustering, TPAMI 2026
            case 'ISC'
                %% 

                 alg_results_dir = fullfile(total_results_dir, alg);
                 if ~exist(alg_results_dir, 'dir')
                     mkdir(alg_results_dir);
                 end

                 fprintf("Processing on "+ data + " dataset!\n")
                 disp('Running algorithm is ISC!')
                 % paramters
                 gamma_set = [1e-3, 1e-2, 1e-1, 1, 10, 100];
                 lambda_set = [1, 1e2,  1e3,  1e4,  1e5,  1e6];

                 opts.Dx = L2_distance_1(X, X);
                 S = initialize_S(X, opts.Dx, opts.n_clusters);
                 predY = SpectralClustering((abs(S')+ abs(S)),opts.n_clusters);
                 F = ISC_ind2vec(predY);
                 
                 clear S predY; % Pre-cleanup before entering the heavy loops
                 
                 isc_time = [];
                 for a_index = 1 : length(gamma_set)
                    for b_index = 1 : length(lambda_set)
                        gamma = gamma_set(a_index);
                        lambda = lambda_set(b_index);

                        if device == "cpu"
                            [C, F, W, time] = isc_cpu(X, F, gamma, lambda, opts);
                        else
                            [C, F, W, time] = isc_gpu(X, F, gamma, lambda, opts);
                        end
                        isc_time = [isc_time time];
                        
                        % Sparse & Isolation
                        % C = sparse(C);
                        W_C1 = abs(C') + abs(C);
                        
                        resultC1 = [];
                        for t = 1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            % predY = kmeans(F',opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end
                        clear W_C1;
                        
                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, b_index} = mresultC1;
                        resultsC1_std{a_index, b_index} = std(resultC1);
                        acc_C1(a_index, b_index) = mresultC1(1)
                        
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(b_index);
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1", "F", "W", "time", '-v7.3');
                        
                        % Garbage Collection
                        % NOTE: Original code passes F to the next iteration. DO NOT clear F here!
                        clear C W time predY result11 resultC1 mresultC1;
                        pause(0.5);
                    end
                end
                isc_ave_time = mean(isc_time);
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "isc_time", '-v7.3');
                clear acc_C1 resultsC1_mean resultsC1_std alg_results_dir alg_data_results_dir filename isc_time F;
                
        end % switch
         
    end % for alg_ind
end % for d_ind
