document.addEventListener("DOMContentLoaded", function() {
    // add links to nav boxes
    boxes = document.querySelectorAll(".toms-nav-container .box, .toms-nav-box");
    boxes.forEach(element => {
        element.addEventListener("click", function() {
            window.location.href = this.getAttribute("data-href");
        })
    });

    // hide the dummy home title
    if (document.getElementById("home")) {
        document.querySelector("#home > h1").style.display = "none"
    }

    let active_el = document.querySelector(".bd-navbar-elements.navbar-nav .nav-item.active")
    if (active_el && active_el.innerText == "Tutorials") {
        if (document.querySelector(".bd-links__title")) {
            document.querySelector(".bd-links__title").innerText = "Other tutorials";
        }
    }
})