document.addEventListener("DOMContentLoaded", function() {
    // add links to nav boxes
    boxes = document.querySelectorAll(".toms-nav-container .box, .toms-nav-box");
    boxes.forEach(element => {
        element.addEventListener("click", function() {
            window.location.href = this.getAttribute("data-href");
        })
    });

    // hide the dummy home/feedback titles
    ["home", "feedback"].forEach(dummy => {
        if (document.getElementById(dummy)) {
            document.querySelector(`#${dummy} > h1`).style.display = "none"
        }
    });

    let active_el = document.querySelector(".bd-navbar-elements.navbar-nav .nav-item.active")
    if (active_el && active_el.innerText == "Tutorials") {
        if (document.querySelector(".bd-links__title")) {
            document.querySelector(".bd-links__title").innerText = "Other tutorials";
        }
    }

    // go through any stderr messages and add tqdm classes as necessary
    document.querySelectorAll(".stderr").forEach(x => {
        if (x.innerText.includes("%") && x.innerText.includes("it/s")) {
            x.classList.add("tqdm")
        }
    })

    // stop forcing the start box to be so large
    let start = document.querySelector(".navbar-header-items__start")
    start.classList.remove("col-lg-3")
    start.classList.add("col-lg")
    let middle = document.querySelector(".navbar-header-items")
    middle.classList.remove("col-lg-9")
    middle.classList.add("col-lg-10")

    // properly hide any 'only-light/dark' figures
    document.querySelectorAll("figure").forEach(el => {
        if (el.querySelector(".only-light")) {
            el.classList.add("only-light");
        }
        if (el.querySelector(".only-dark")) {
            el.classList.add("only-dark");
        }
    })
})