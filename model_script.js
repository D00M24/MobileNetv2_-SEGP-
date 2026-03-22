async function diagnoseDogPosture(imageElements) {
    if (imageElements.length !== 4) {
        console.error("❌ ERROR: Exactly 4 images are required!");
        return;
    }

    console.log("=========================================");
    console.log("🚀 STARTING AI DIAGNOSIS DEBUG RUN");
    console.log("=========================================");

    const model = await tf.loadGraphModel('./model/model.json'); 
    console.log("✅ Model loaded successfully from memory.");
    
    let redFlags = 0;
    let individualScores = [];

    for (let i = 0; i < imageElements.length; i++) {
        console.log(`\n--- Processing Image ${i + 1} of 4 ---`);
        
        const tensor = tf.tidy(() => {
            let imgTensor = tf.browser.fromPixels(imageElements[i]);
            let resized = tf.image.resizeBilinear(imgTensor, [224, 224]);
            let normalized = resized.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1.0));
            let finalTensor = normalized.expandDims(0); 
            
            // DEBUG: Check if the shape is exactly [1, 224, 224, 3]
            console.log(`🖼️ Image ${i + 1} Tensor Shape: [${finalTensor.shape}]`);
            return finalTensor;
        });

        // Run the prediction
        const prediction = await model.predict(tensor).data();
        const score = prediction[0]; 
        individualScores.push(score);

        // DEBUG: Print the exact raw decimal the AI predicted
        console.log(`🧠 RAW AI SCORE: ${score}`);

        if (score >= 0.5) {
            console.log(`🚩 RED FLAG TRIGGERED! (Score >= 0.5)`);
            redFlags += 1;
        } else {
            console.log(`✅ NORMAL POSTURE (Score < 0.5)`);
        }

        tf.dispose(tensor); 
    }

    const confidence = (redFlags / 4) * 100;
    const diagnosis = redFlags >= 2 ? "Abnormal" : "Normal";

    console.log("\n=========================================");
    console.log("📊 FINAL ENSEMBLE RESULTS");
    console.log(`Raw Array of Scores: [${individualScores}]`);
    console.log(`Total Red Flags: ${redFlags} out of 4`);
    console.log(`Final Diagnosis: ${diagnosis}`);
    console.log(`Confidence: ${confidence}%`);
    console.log("=========================================\n");

    return {
        diagnosis: diagnosis,
        confidence: confidence,
        redFlags: redFlags
    };
}