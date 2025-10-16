// ✅ Expose DisplayMessage to Python
eel.expose(DisplayMessage);
function DisplayMessage(message) {
    console.log("From Python:", message);
    $(".siri-message").text(message);
    
    $('.siri-message').textillate('stop'); 

    $('.siri-message').textillate({
        in: { effect: 'fadeInUp', sync: true },
        out: { effect: 'fadeOutUp', sync: true },
        callback: function () {
            console.log("✅ Text animation complete");
            // Only notify Python when NOT in continuous mode, 
            // as Python handles the looping itself now.
            if (!is_continuous_mode) {
                 eel.display_done(); 
            }
        }
    });

    $('.siri-message').textillate('start');
}

// 🌐 NEW GLOBAL STATE: To track if we are in the continuous listening loop
let is_continuous_mode = false;

// ✅ Expose showHood to Python (Used for text mode fallback)
eel.expose(showHood);
function showHood() {
    is_continuous_mode = false;
    $("#Oval").show();
    $("#SiriWave").hide();
}

// ✅ NEW: Function for Python to control the Mic/Hood state
eel.expose(setMicState);
function setMicState(state) {
    if (state === "continuous") {
        is_continuous_mode = true;
        // Keep in SiriWave mode and the loop handles the rest
    } else if (state === "idle") {
        // Conversation ended by Python (e.g., user said "quit")
        showHood();
    }
}


// ✅ Single Mic button handler
$("#MicBtn").click(function () {
    const micSound = document.getElementById("micSound");
    if (micSound) micSound.play();
    
    // 1. Set UI for listening
    $("#Oval").hide();
    $("#SiriWave").show();
    $("#siriwave").empty();

    new SiriWave({
        container: document.getElementById('siriwave'),
        width: 600,
        height: 200,
        style: 'ios9',
        amplitude: 1,
        speed: 0.2,
        autostart: true,
    });
    
    // 2. Notify Python to play sound and START THE CONVERSATION LOOP
    if (typeof eel !== "undefined" && typeof eel.playAssistantSound === "function") {
      eel.playAssistantSound();
    }
    
    // 3. 🌟 CRITICAL: Call the new start_conversation function
    eel.start_conversation();
});