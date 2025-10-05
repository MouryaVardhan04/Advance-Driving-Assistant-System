// ✅ Expose DisplayMessage to Python
eel.expose(DisplayMessage);
function DisplayMessage(message) {
    console.log("From Python:", message);
    $(".siri-message").text(message);
    
    $('.siri-message').textillate({
        in: { effect: 'fadeInUp', sync: true },
        out: { effect: 'fadeOutUp', sync: true },
        callback: function () {
            console.log("✅ Text animation complete");
            eel.display_done();  // Notify Python
        }
    });

    $('.siri-message').textillate('start');
}

// ✅ Expose showHood to Python
eel.expose(showHood);
function showHood() {
    $("#Oval").show();
    $("#SiriWave").hide();
}

// ✅ Single Mic button handler
$("#MicBtn").click(function () {
    const micSound = document.getElementById("micSound");
    if (micSound) micSound.play();

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

    eel.takecommand()();
});
