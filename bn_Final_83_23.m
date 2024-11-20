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
    T_bnMaligno{i} = 'Maligno';
end

for i = 1:636
    T_Benigno(i, :) = [1, 0];
    T_bnBenigno{i} = 'Benigno';
end

% Etiquetas
input = [Train_Maligno; Train_Benigno];
Target = [T_Maligno; T_Benigno];
T_bn = [T_bnMaligno; T_bnBenigno];
T_bn = cellstr(T_bn);

% Entrenamiento del modelo Naive Bayes
NBModel = fitcnb(input(1:1002, :), T_bn(1:1002));

BDVAL_Benigno = readtable('val_benigno.csv');
BDVAL_Maligno = readtable('val_maligno.csv');
Bien = table2array(BDVAL_Maligno);
Mal = table2array(BDVAL_Benigno);

% Predicción y métricas
TP = 0; TN = 0; FN = 0; FP = 0;

% Predicción para datos benignos
for i = 1:size(Mal, 1)
    res = string(predict(NBModel, Mal(i, :)));
    if res == "Benigno"
        TP = TP + 1;
    else
        FP = FP + 1;
    end
end

% Predicción para datos malignos
for i = 1:size(Bien, 1)
    res = string(predict(NBModel, Bien(i, :)));
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
