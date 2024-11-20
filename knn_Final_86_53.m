clc; clear all; close all;

% Leer la tabla de datos
BD = readtable('base_datos_cancer_2.csv');
B = 1;
M = 1;

% Inicialización de matrices
Benigno = [];
Maligno = [];

% Separación de datos en Benigno y Maligno
for i = 1:1707
    if BD.Var31(i) == 0
        Benigno(B, :) = BD{i, 1:31};
        B = B + 1;
    else
        Maligno(M, :) = BD{i, 1:31};
        M = M + 1;
    end
end

% División de datos en entrenamiento y prueba
c = 1; c2 = 1;
for i = 1:636
    if (i < 637)
        Train_Benigno(c, :) = Benigno(i, :);
        c = c + 1;
    else
        Test_Benigno(c2, :) = Benigno(i, :);
        c2 = c2 + 1;
    end
end

c = 1; c2 = 1;
for i = 1:636
    if (i < 637)
        Train_Maligno(c, :) = Maligno(i, :);
        c = c + 1;
    else
        Test_Maligno(c2, :) = Maligno(i, :);
        c2 = c2 + 1;
    end
end

% Etiquetas de clase
for i = 1:636
    T_Maligno(i, :) = [0, 1];
    T_knnMaligno{i} = 'Maligno';
end

for i = 1:636
    T_Benigno(i, :) = [1, 0];
    T_knnBenigno{i} = 'Benigno';
end

% Etiquetas
input = [Train_Maligno; Train_Benigno];
Target = [T_Maligno; T_Benigno];
T_knn = [T_knnMaligno; T_knnBenigno];
T_knn = cellstr(T_knn);

% Definir el rango de hiperparámetros
kernelScales = logspace(-3, 3, 7);
boxConstraints = logspace(-3, 3, 7);

% Inicializar variables para almacenar los mejores resultados
bestAccuracy = 0;
bestKernelScale = 1;
bestBoxConstraint = 1;

% Búsqueda en cuadrícula
for ks = kernelScales
    for bc = boxConstraints
        % Estandarizar los datos manualmente si es necesario
        inputStandardized = (input - mean(input)) ./ std(input);
        
        Mdl = fitcknn(inputStandardized(1:1270, :), T_knn(1:1270), 'NumNeighbors', 3, ...
            'NSMethod', 'exhaustive', 'Distance', 'minkowski');

        % Validación cruzada
        CVMDLModel = crossval(Mdl, 'KFold', 5);
        accuracy = 1 - kfoldLoss(CVMDLModel);

        % Actualizar los mejores hiperparámetros
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestKernelScale = ks;
        end
    end
end

% Entrenar el modelo final con los mejores hiperparámetros
inputStandardized = (input - mean(input)) ./ std(input);
Mdl = fitcknn(inputStandardized(1:1270, :), T_knn(1:1270), 'NumNeighbors', 3, ...
    'NSMethod', 'exhaustive', 'Distance', 'minkowski');

BDVAL_Benigno = readtable('val_benigno.csv');
BDVAL_Maligno = readtable('val_maligno.csv');
Bien = table2array(BDVAL_Maligno);
Mal = table2array(BDVAL_Benigno);

% Predicción y métricas
TP = 0; TN = 0; FN = 0; FP = 0;

% Predicción para datos benignos
for i = 1:size(Mal, 1)
    res = string(predict(Mdl, Mal(i, :)));
    if res == "Benigno"
        TP = TP + 1;
    else
        FP = FP + 1;
    end
end

% Predicción para datos malignos
for i = 1:size(Bien, 1)
    res = string(predict(Mdl, Bien(i, :)));
    if res == "Maligno"
        TN = TN + 1;
    else
        FN = FN + 1;
    end
end

% Cálculo de métricas
Exac = ((TP + TN) / (TP + TN + FP + FN)) * 100;
Pre = (TP / (FP + TP)) * 100;
Rec = (TP / (TP + FN)) * 100;
F1 = 2 * ((Pre * Rec) / (Pre + Rec));

% Mostrar resultados
fprintf('Exactitud: %.2f%%\n', Exac);
fprintf('Precisión: %.2f%%\n', Pre);
fprintf('Recall: %.2f%%\n', Rec);
fprintf('F1 Score: %.2f%%\n', F1);
