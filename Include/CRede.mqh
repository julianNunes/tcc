//+------------------------------------------------------------------+
//|                                                        CRede.mqh |
//|                                                     Julian Nunes |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Julian Nunes"
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Arrays\ArrayDouble.mqh>
#include <Arrays\ArrayInt.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Tcc\CNeuronio.mqh>
#include <Tcc\CCamada.mqh>
#include <Tcc\CArrayCamada.mqh>


/**
 * Classe da Rede Neural
 */
class CRede
{
    public:
        CRede(const CArrayInt *topology);
        ~CRede() { layers.Clear(); };

        bool feedForward(CArrayDouble *inputVals);
        bool backProp(CArrayDouble *targetVals);
        bool getResults(CArrayDouble *resultVals) const;
        double getRecentAverageError() const { return recentAverageError; }

        bool Save(
            const string file_name,
            double error,
            double undefine,
            double forecast,
            datetime time,
            bool common=true
        );
        bool Load(
            const string file_name,
            double &error,
            double &undefine,
            double &forecast,
            datetime &time,
            bool common=true
        );

        static double recentAverageSmoothingFactor;

    private:
        CArrayCamada layers;
        double recentAverageError;
};

/**
 * armazenar o erro médio
 * Fator de suavização
 */
double CRede::recentAverageSmoothingFactor = 10000.0; // Number of training samples to average over

/**
 * Um ponteiro para a matriz de dados do tipo int é passada nos parâmetros do construtor da classe.
 * O número de elementos na matriz indica o número de camadas, enquanto cada elemento da matriz contém o número de neurônios na camada apropriada
 *
 */
CRede::CRede(const CArrayInt *topology)
{
    if (CheckPointer(topology) == POINTER_INVALID) { return; }

    int numLayers = topology.Total();
    // Print("numLayers " + IntegerToString(numLayers));

    //  Um valor igual a zero de conexões de saída é especificado para o nível de saída.
    for (int layerNum = 0; layerNum < numLayers; layerNum++) {
        uint numOutputs = (layerNum == (numLayers - 1) ? 0 : topology.At(layerNum + 1));
        // Criando camadas da rede
        if (!layers.CreateElement(topology.At(layerNum), numOutputs)) {
            Print("CRede: problema em CreateElement");
            return;
        }
    }
}


/**
 * Usado para calcular o valor da rede neural
 * *inputVals - matriz de valores de entrada, com base nos quais os valores resultantes da rede neural serão calculados.
 */
bool CRede::feedForward(CArrayDouble *inputVals)
{
    // Validando do ponteiro de recebimento
    if (CheckPointer(inputVals) == POINTER_INVALID) {
        Print("CRede::feedForward - inputVals POINTER_INVALID");
        return false;
    }

    // Validando camada zero de nossa rede
    CCamada *Layer = layers.At(0);

    if (CheckPointer(Layer) == POINTER_INVALID) {
        Print("CRede::feedForward - Layer POINTER_INVALID");
        return false;
    }

    int total = inputVals.Total();

    if (total != Layer.Total()) {
        Print("total != Layer.Total() - 1");
        return false;
    }

    // definimos os valores iniciais recebidos com os valores resultantes dos neurônios da camada zero
    for (int i = 0; i < total && !IsStopped(); i++) {
        CNeuronio *neuron = Layer.At(i);
        neuron.setOutputVal(inputVals.At(i));
    }

    total = layers.Total();
    // recálculo em fases dos valores resultantes dos neurônios em toda a rede neural,
    // da primeira camada oculta aos neurônios de saída.
    for (int layerNum = 1; layerNum < total && !IsStopped(); layerNum++) {
        CArrayObj *prevLayer = layers.At(layerNum - 1);
        CArrayObj *currLayer = layers.At(layerNum);
        int t = currLayer.Total();

        for (int n = 0; n < t && !IsStopped(); n++) {
            CNeuronio *neuron = currLayer.At(n);
            neuron.feedForward(prevLayer);
        }
    }

    return true;
}

/**
 *
 */
bool CRede::getResults(CArrayDouble *resultVals) const
{
    if (CheckPointer(resultVals) == POINTER_INVALID) { resultVals = new CArrayDouble(); }

    resultVals.Clear();
    CArrayObj *Layer = layers.At(layers.Total() - 1);

    if (CheckPointer(Layer) == POINTER_INVALID) {
        Print("CRede::getResults - Layer POINTER_INVALID");
        return false;
    }

    int total = Layer.Total();

    for (int n = 0; n < total; n++) {
        CNeuronio *neuron = Layer.At(n);
        resultVals.Add(neuron.getOutputVal());
    }

    return true;
}

/**
 * *targetVals - matriz de valores de referência de parâmetros
 * O processo de aprendizado da rede neural é implementado no método backProp
 *
 */
bool CRede::backProp(CArrayDouble *targetVals)
{
    if (CheckPointer(targetVals) == POINTER_INVALID) {
        Print("CRede::backProp - targetVals POINTER_INVALID");
        return false;
    }

    CArrayObj *outputLayer = layers.At(layers.Total() - 1);

    if (CheckPointer(outputLayer) == POINTER_INVALID) {
        Print("CRede::backProp - outputLayer POINTER_INVALID");
        return false;
    }

    double error = 0.0;
    double delta = 0.0;
    int total = outputLayer.Total();

    for (int n = 0; n < total && !IsStopped(); n++) {
        CNeuronio *neuron = outputLayer.At(n);
        delta = targetVals[n] - neuron.getOutputVal();
        error += delta * delta;
    }

    error /= total;
    // EXISTE A VERSÃO SEM A RAIZ QUADRADA
    error = sqrt(error);

    // calculamos o erro quadrático médio da camada resultante
    recentAverageError += (error - recentAverageError) / recentAverageSmoothingFactor;

    // recalculamos os gradientes dos neurônios em todas a camada
    // Calculo do gradiente para a camada de saída
    for (int n = 0; n < total && !IsStopped(); n++) {
        CNeuronio *neuron = outputLayer.At(n);
        neuron.calcOutputGradients(targetVals.At(n));
    }

    // recalculamos os gradientes dos neurônios da camada escondida
    for (int layerNum = (layers.Total() - 2); layerNum >= 0; layerNum--) {
        CArrayObj *hiddenLayer = layers.At(layerNum);
        total = hiddenLayer.Total();
        CArrayObj *nextLayer = layers.At(layerNum + 1);

        for (int n = 0; n < total && !IsStopped(); n++) {
            CNeuronio *neuron = hiddenLayer.At(n);
            neuron.calcHiddenGradients(nextLayer);
        }
    }

    // atualizamos os pesos das conexões entre os neurônios com base nos gradientes calculados anteriormente
    for (int layerNum = layers.Total() - 1; layerNum > 0; layerNum--) {
        CArrayObj *layer = layers.At(layerNum);
        total = layer.Total();
        CArrayObj *prevLayer = layers.At(layerNum - 1);

        for (int n = 0; n < total && !IsStopped(); n++) {
            CNeuronio *neuron = layer.At(n);
            neuron.updateInputWeights(prevLayer);
        }
    }

    return true;
}


/**
 *
 */
bool CRede::Save(
    const string file_name,
    double loop_err,
    double undefine_p,
    double forecast_er,
    datetime time,
    bool common=true
) {
//    if (MQLInfoInteger(MQL_OPTIMIZATION)
//         || MQLInfoInteger(MQL_TESTER)
//         || MQLInfoInteger(MQL_FORWARD)
//         || MQLInfoInteger(MQL_OPTIMIZATION)
//     ) {
//         return true;
//     }

    if (file_name == NULL) {
        Print("CRede:save() - SEM NOME PARA ARQUIVO");
        return false;
    }

    int handle = FileOpen(file_name, FILE_BIN | FILE_WRITE | FILE_COMMON);

    if (handle == INVALID_HANDLE) {
        Print("CRede:save() - INVALID_HANDLE FILE");
        return false;
    }

    if (FileWriteDouble(handle, loop_err) <= 0
        || FileWriteDouble(handle, undefine_p) <= 0
        || FileWriteDouble(handle, forecast_er) <= 0
        || FileWriteLong(handle,(long) time) <= 0
    ) {
        FileClose(handle);
        Print("CRede:save() - ERROR DADOs");
        return false;
    }

    bool result = layers.Save(handle);

    FileFlush(handle);
    FileClose(handle);

    if (result) {
        Print("CRede:save() - DEU CERTO");
    } else {
        Print("CRede:save() - ERRO");
    }

    return result;
}

/**
 *
 */
bool CRede::Load(
    const string file_name,
    double &loop_err,
    double &undefine_p,
    double &forecast_er,
    datetime &time,
    bool common=true
) {
    // if (MQLInfoInteger(MQL_OPTIMIZATION)
    //     || MQLInfoInteger(MQL_TESTER)
    //     || MQLInfoInteger(MQL_FORWARD)
    //     || MQLInfoInteger(MQL_OPTIMIZATION)
    // ) {
    //     return false;
    // }

    if (file_name == NULL) {
        Print("CRede:Load() - SEM NOME PARA ARQUIVO");
        return false;
    }

    int handle = FileOpen(file_name, FILE_BIN | FILE_READ | FILE_COMMON);
    if (handle == INVALID_HANDLE) {
        Print("CRede:Load() - INVALID_HANDLE FILE");
        return false;
    }

    loop_err = FileReadDouble(handle);
    undefine_p = FileReadDouble(handle);
    forecast_er = FileReadDouble(handle);
    time = (datetime) FileReadLong(handle);

    layers.Clear();
    int i = 0, num;

    //--- check
    //--- read and check start marker - 0xFFFFFFFFFFFFFFFF
    if(FileReadLong(handle) == -1) {
        //--- read and check array type
        if(FileReadInteger(handle, INT_VALUE) != layers.Type()) {
            Print("CRede::Load - read and check array type");
            return false;
        }
    } else {
        Print("CRede::Load() -  read and check start marker - 0xFFFFFFFFFFFFFFFF");
        return false;
    }

    //--- read array length
    num = FileReadInteger(handle, INT_VALUE);
    //--- read array
    if(num != 0) {
        for(i = 0; i < num; i++) {
            //--- create new element
            CCamada *Layer = new CCamada();

            if (!Layer.Load(handle)) {
                Print("CRede::Load() -  Load CCamada");
                break;
            }

            if (!layers.Add(Layer)) {
                Print("CRede::Load() -  ADD CCamada");
                break;
            }
        }
    }

    FileClose(handle);
    //--- result
    return (layers.Total() == num);
}

