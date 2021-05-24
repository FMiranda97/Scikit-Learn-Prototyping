Este repositório contém um aplicação web desenvolvida em django com o objetivo de facilitiar o trabalho experimental em problemas de classificação.
Os ficheiros relevantes respeitando à aplicação web são:
    - todos aqueles presentes na subdirectoria RP2021
    - os ficheiros feature_selection.py, model_creation.py e data_initialization.py
    - o ficheiro MDC.py que contém uma implementação de um modelo Minimum Distance Classifier retirado de "https://github.com/RomuloDrumond/Minimum-Distance-Classifier/blob/master/Minimum%20Distance%20Classifier%20(MDC).ipynb"

O script data_extration.py faz a junção dos diferentes CSV's num único
e por fim o script "csv transformer.py" faz o tratamento de valores em falta e mapeamento de valores categóricos em valores numéricos.

Além dos ficheiros referidos, a subdirectoria "jupyter notebooks" contém alguns scripts com experiências executadas no contexto do trabalho mas que não foram exploradas o suficiente para exposição no relatório.