const predictClassification = require('../services/inferenceService');
const crypto = require('crypto');
const storeData = require('../services/storeData');
 
async function postPredictHandler(request, h) {
  const { image } = request.payload;
  const { model } = request.server.app;
  const { confidenceScore, label, explanation, suggestion } = await predictClassification(model, image);
  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();
  const data = {
    "id": id,
    "result": label,
    "suggestion": suggestion,
    "createdAt": createdAt
  }
  
  await storeData(id, data);
  const response = h.response({
    status: 'success',
    //message: confidenceScore > 99 ? 'Model is predicted successfully.' : 'Model is predicted successfully but it is not a skin cancer',
    message: 'Model is predicted successfully',
    data
  })
  response.code(201);
  return response;
}
 
module.exports = postPredictHandler;