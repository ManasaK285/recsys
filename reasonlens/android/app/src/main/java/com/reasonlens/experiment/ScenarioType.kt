package com.reasonlens.experiment

/** Mirrors backend/app/config.py::APP_TYPES exactly. */
enum class ScenarioType(val key: String, val displayName: String, val description: String) {
    RIDESHARE(
        "rideshare",
        "QuickRide",
        "A ride-hailing app screen showing a map and a \"Request ride\" button."
    ),
    FOOD_DELIVERY(
        "food_delivery",
        "TastyGo",
        "A food delivery app screen showing nearby restaurants."
    ),
    LOCAL_NEWS(
        "local_news",
        "AreaWire",
        "A local news app screen showing a headline feed."
    ),
    WALLPAPER(
        "wallpaper",
        "Wallify",
        "A wallpaper app screen showing a gallery of backgrounds."
    );

    companion object {
        fun fromKey(key: String): ScenarioType = entries.first { it.key == key }
    }
}
