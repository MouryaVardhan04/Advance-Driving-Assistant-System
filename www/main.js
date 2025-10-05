// main.js

// 💡 Define showHood BEFORE document.ready so Python can call it immediately
eel.expose(showHood);
function showHood() {
    console.log("Python called showHood: Switching to Hood/Oval view.");
    // 💡 FIX: Hide SiriWave and show Oval
    $("#SiriWave").hide(); 
    $("#Oval").show();   
    
    // Reset buttons
    $("#MicBtn").show();
    $("#SendBtn").hide();
}


$(document).ready(function () {

    eel.init()()

    // Ensure initial state is correct (Oval visible, SiriWave hidden)
    $("#Oval").show();
    $("#SiriWave").hide();

    $('.text').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "bounceIn",
        },
        out: {
            effect: "bounceOut",
        },

    });

    // Siri configuration
    new SiriWave({
      container: document.getElementById('siriwave'),
      width: 600,
      height: 200,
      style: 'ios9',
      amplitude: 1,
      speed: 0.2,
      autostart: true,
    });

    // Siri message animation
    $('.siri-message').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "fadeInUp",
            sync: true,
        },
        out: {
            effect: "fadeOutUp",
            sync: true,
        },

    });

    // mic button click event
    $("#MicBtn").click(function () { 
        // 💡 FIX: Use .show()/.hide() for consistency
        $("#Oval").hide();
        $("#SiriWave").show();
        
        eel.playAssistantSound();
        eel.takecommand()();
    });


    function doc_keyUp(e) {
        // Mac Command+J (metaKey for Command key on Mac)
        if (e.key === 'j' && e.metaKey) {
            $("#Oval").hide();
            $("#SiriWave").show();
            eel.playAssistantSound();
            eel.takecommand()();
        }
    }
    document.addEventListener('keyup', doc_keyUp, false);

    // to play assistant 
    function PlayAssistant(message) {

        if (message != "") {

            // Switch to SiriWave display when manually submitting command
            $("#Oval").hide();
            $("#SiriWave").show();
            
            eel.allCommands(message);
            
            $("#chatbox").val("")
            ShowHideButton("")
        }

    }

    // toogle fucntion to hide and display mic and send button 
    function ShowHideButton(message) {
        if (message.length === 0) {
            $("#MicBtn").show(); // Show mic
            $("#SendBtn").hide(); // Hide send
        }
        else {
            $("#MicBtn").hide(); // Hide mic
            $("#SendBtn").show(); // Show send
        }
    }

    // key up event handler on text box
    $("#chatbox").keyup(function () {
        let message = $("#chatbox").val();
        ShowHideButton(message)
    });
    
    // send button event handler
    $("#SendBtn").click(function () {
        let message = $("#chatbox").val()
        PlayAssistant(message)
    });
    

    // enter press event handler on chat box
    $("#chatbox").keypress(function (e) {
        key = e.which;
        if (key === 13) {
            let message = $("#chatbox").val()
            PlayAssistant(message)
        }
    });
});