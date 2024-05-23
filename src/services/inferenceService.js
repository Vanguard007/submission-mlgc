const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
 
async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat()
        
        const classes = ['Cancer', 'Non-cancer'];

        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;
        //const confidenceScore = score[0];
 
        const classResult = tf.argMax(prediction, 1).dataSync()[0];
        //const label = classes[classResult];
        const label = confidenceScore >= 50 ? classes[classResult] : 'Non-cancer';
 
        let explanation, suggestion;
 
        if (label === 'Cancer') {
            //explanation = "Melanocytic nevus adalah kondisi di mana permukaan kulit memiliki bercak warna yang berasal dari sel-sel melanosit, yakni pembentukan warna kulit dan rambut."
            suggestion = "Segera periksa ke dokter!"
          } else if(label === 'Non-cancer'){
            //explanation = "Squamous cell carcinoma adalah jenis kanker kulit yang umum dijumpai. Penyakit ini sering tumbuh pada bagian-bagian tubuh yang sering terkena sinar UV."
            suggestion = "Anda sehat!"
          }
 
        return { confidenceScore, label, explanation, suggestion };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan input: ${error.message}`)
    }
}
 
module.exports = predictClassification;