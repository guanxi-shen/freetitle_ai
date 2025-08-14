"""Cinematic style extension for the script agent

Appended to script agent prompt to produce structured cinematic style directives
alongside the shooting script. Covers shot language, pacing, mood, composition,
texture, lighting, lens behavior, color science, sound design cues, and editorial rhythm.
"""

# Structured style library for programmatic access (e.g., style picker UI, agent routing)
CINEMATIC_STYLE_LIBRARY = {
    "wong_kar_wai": {
        "label": "Wong Kar Wai",
        "keywords": [
            "saturated film colors", "neon bokeh nights", "step-printing motion blur",
            "handheld drift", "rain-soaked streets", "intimate close-ups",
            "moody urban romance", "fragmented time montage", "cigarette haze halation"
        ]
    },
    "film_noir": {
        "label": "Film Noir",
        "keywords": [
            "hard chiaroscuro", "venetian-blind shadows", "wet asphalt reflections",
            "tungsten streetlamps", "smoky interiors", "high-contrast monochrome",
            "oblique dutch angles", "slow push-ins", "shadow-silhouette blocking"
        ]
    },
    "neon_cyberpunk_noir": {
        "label": "Neon Cyberpunk Noir",
        "keywords": [
            "magenta-cyan LEDs", "holographic signage glow", "anamorphic streak flares",
            "chrome raincoat highlights", "wet neon puddles", "deep night fog",
            "aggressive contrast", "glitchy jump cuts", "synthwave pulse"
        ]
    },
    "epic_grounded_realism": {
        "label": "Epic Grounded Realism",
        "keywords": [
            "large-scale wide framing", "practical lighting realism", "low-saturation palette",
            "heavy bass hit cues", "clockwork cross-cutting", "aerial establishing momentum",
            "tactile sharp texture", "restrained handheld", "looming wides"
        ]
    },
    "storybook_symmetry": {
        "label": "Storybook Symmetry",
        "keywords": [
            "perfect center framing", "pastel color blocking", "snap zooms",
            "lateral dolly moves", "diorama-like sets", "deadpan beat timing",
            "crisp daylight fill", "graphic title cards", "rhythmic chapter edits"
        ]
    },
    "surreal_dread": {
        "label": "Surreal Dread",
        "keywords": [
            "uncanny negative space", "low hum ambience", "sodium-vapor sickly light",
            "slow creeping zoom", "abrupt silence-to-noise spikes", "dream-logic continuity breaks",
            "macro texture inserts", "uneasy close-ups"
        ]
    },
    "70s_gritty_urban": {
        "label": "70s Gritty Urban",
        "keywords": [
            "dirty street realism", "long-lens compression", "handheld verite wobble",
            "cigarette-yellow tungsten", "heavy film grain", "muted earth tones",
            "imperfect focus pulls", "harsh practicals", "raw location sound"
        ]
    },
    "90s_vhs_nostalgia": {
        "label": "90s VHS Nostalgia",
        "keywords": [
            "VHS tracking noise", "chroma bleed", "timecode overlays",
            "soft interlaced blur", "tape warble", "blown highlights",
            "fluorescent cast", "sloppy zoom cam", "abrupt stop-start cuts"
        ]
    },
    "y2k_glossy_pop": {
        "label": "Y2K Glossy Pop",
        "keywords": [
            "high-key shine", "metallic highlights", "lens bloom",
            "chromatic aberration edges", "fast whip pans", "UI overlay graphics",
            "liquid gradients", "strobe edits", "bubblegum neon palette"
        ]
    },
    "music_video_hypercut": {
        "label": "Music Video Hypercut",
        "keywords": [
            "micro-beat cutting", "whip-pan transitions", "speed ramps",
            "strobe lighting", "camera flash pops", "kinetic handheld",
            "smash match cuts", "glitch frames", "percussive visual rhythm"
        ]
    },
    "luxury_fashion_editorial": {
        "label": "Luxury Fashion Editorial",
        "keywords": [
            "controlled softbox lighting", "highlight roll-off", "shallow DOF glam close-ups",
            "slow-motion fabric drift", "minimal palette", "negative space composition",
            "sharp silhouettes", "editorial pacing", "clean typography cards"
        ]
    },
    "warm_wonder_animation": {
        "label": "Warm Wonder Animation",
        "keywords": [
            "watercolor skies", "soft rim light", "gentle wind motion",
            "warm pastoral palette", "cozy interior glow", "hand-painted texture",
            "wide tranquil establishes", "small character gestures", "lyrical pacing"
        ]
    },
    "neo_anime_intensity": {
        "label": "Neo-Anime Intensity",
        "keywords": [
            "hard-edged shadows", "neon city signage", "extreme angle perspective",
            "rapid insert cuts", "high-contrast night palette", "dramatic eye close-ups",
            "kinetic camera sweeps", "impact-frame flashes"
        ]
    },
    "3d_family_warmth": {
        "label": "3D Family Warmth",
        "keywords": [
            "soft global illumination", "expressive face close-ups", "clean depth-of-field",
            "rich color harmony", "bouncy camera easing", "comedic timing beats",
            "bright specular eyes", "cozy bounce light"
        ]
    },
    "stop_motion_handmade": {
        "label": "Stop-Motion Handmade",
        "keywords": [
            "tactile clay/felt textures", "slight frame jitter", "miniature set depth",
            "practical tiny lights", "handcrafted props", "warm vignette",
            "whimsical cut timing", "charming imperfections"
        ]
    },
    "analog_horror": {
        "label": "Analog Horror",
        "keywords": [
            "CRT scanlines", "crushed blacks", "overexposed hotspots",
            "found-footage shakiness", "warning text overlays", "audio dropouts",
            "unsettling still frames", "surveillance zoom", "corrupted glitches"
        ]
    },
    "epic_high_fantasy": {
        "label": "Epic High Fantasy",
        "keywords": [
            "golden-hour god rays", "sweeping crane reveals", "wind-swept silhouettes",
            "ornate costume detail", "painterly haze", "heroic wides",
            "choral swell pacing", "majestic slow push-ins"
        ]
    },
    "minimalist_arthouse_realism": {
        "label": "Minimalist Arthouse Realism",
        "keywords": [
            "natural window light", "long static takes", "quiet room tone",
            "muted neutrals", "sparse blocking", "observational distance",
            "subtle handheld micro-movement", "restrained cuts", "candid imperfections"
        ]
    },
}


# Appended to script agent prompt to enable cinematic style output
CINEMATIC_STYLE_INSTRUCTIONS = """
========================================
CINEMATIC STYLE DIRECTIVES
========================================

In addition to the shooting script, distill the video's overall cinematic taste into a structured style package. This package guides downstream storyboard, video generation, and editing agents on the audiovisual language of the piece.

Style distillation means identifying what makes the look emotionally and visually recognizable, then expressing it as concrete, actionable directives -- not vague adjectives.

WHAT TO PRODUCE:
1. Interpret the user's desired vibe (romantic, tense, dreamlike, luxurious, gritty, etc.) into specific cinematic language.
2. Produce concrete style directives executable in cinematography + edit + color + sound.
3. Output keyword bundles and micro-rules that downstream agents can consume directly.
4. All choices must reinforce one coherent directorial vision consistent with your script.

CINEMATIC STYLE OUTPUT (include as "cinematic_style" object in your JSON response):

"cinematic_style": {
    "style_thesis": "1-2 sentences: the emotional core + the cinematic language used to express it",
    "stylizer_keywords": ["8-14 high-signal concrete keywords: color, light, lens artifacts, camera movement, edit rhythm, texture, environment, performance framing"],
    "micro_rules": ["6-10 actionable rules, e.g.: favor intimate close-ups with shallow depth, use step-printing on emotional peaks, cut on percussion, use wet reflections at night"],
    "avoid_list": ["3-6 items to prevent generic drift, e.g.: overly clean digital sharpness, random lens flares, inconsistent grade, over-cutting"],
    "prompt_injection_block": "Compact paragraph that downstream agents append to their generation prompts"
}

CONSTRAINTS:
- Be specific: name camera behavior, lens artifacts, light sources, palette behavior, editing rhythms.
- Prioritize recognizable cinematic cues over gear jargon.
- Write as descriptive cinematic language, not "copy X director."
- Keywords are guidance, not a limit -- extrapolate and invent additional coherent keywords as needed.
- Internal consistency: color + movement + pacing + texture must feel like one intentional look.
- The cinematic_style must be consistent with your style_guide, shot descriptions, and audio_design.

REFERENCE STYLE KEYWORD LIBRARY:

Use as reference anchors. Mix, adapt, and extrapolate beyond this list.

1. Wong Kar Wai style:
   saturated film colors, neon bokeh nights, step-printing motion blur, handheld drift, rain-soaked streets, intimate close-ups, moody urban romance, fragmented time montage, cigarette haze halation.

2. Film Noir style:
   hard chiaroscuro, venetian-blind shadows, wet asphalt reflections, tungsten streetlamps, smoky interiors, high-contrast monochrome, oblique dutch angles, slow push-ins, shadow-silhouette blocking.

3. Neon Cyberpunk Noir style:
   magenta-cyan LEDs, holographic signage glow, anamorphic streak flares, chrome raincoat highlights, wet neon puddles, deep night fog, aggressive contrast, glitchy jump cuts, synthwave pulse.

4. Epic Grounded Realism style:
   large-scale wide framing, practical lighting realism, low-saturation palette, heavy bass hit cues, clockwork cross-cutting, aerial establishing momentum, tactile sharp texture, restrained handheld, looming wides.

5. Storybook Symmetry style:
   perfect center framing, pastel color blocking, snap zooms, lateral dolly moves, diorama-like sets, deadpan beat timing, crisp daylight fill, graphic title cards, rhythmic chapter edits.

6. Surreal Dread style:
   uncanny negative space, low hum ambience, sodium-vapor sickly light, slow creeping zoom, abrupt silence-to-noise spikes, dream-logic continuity breaks, macro texture inserts, uneasy close-ups.

7. 70s Gritty Urban style:
   dirty street realism, long-lens compression, handheld verite wobble, cigarette-yellow tungsten, heavy film grain, muted earth tones, imperfect focus pulls, harsh practicals, raw location sound.

8. 90s VHS Nostalgia style:
   VHS tracking noise, chroma bleed, timecode overlays, soft interlaced blur, tape warble, blown highlights, fluorescent cast, sloppy zoom cam, abrupt stop-start cuts.

9. Y2K Glossy Pop style:
   high-key shine, metallic highlights, lens bloom, chromatic aberration edges, fast whip pans, UI overlay graphics, liquid gradients, strobe edits, bubblegum neon palette.

10. Music Video Hypercut style:
    micro-beat cutting, whip-pan transitions, speed ramps, strobe lighting, camera flash pops, kinetic handheld, smash match cuts, glitch frames, percussive visual rhythm.

11. Luxury Fashion Editorial style:
    controlled softbox lighting, highlight roll-off, shallow DOF glam close-ups, slow-motion fabric drift, minimal palette, negative space composition, sharp silhouettes, editorial pacing, clean typography cards.

12. Warm Wonder Animation style:
    watercolor skies, soft rim light, gentle wind motion, warm pastoral palette, cozy interior glow, hand-painted texture, wide tranquil establishes, small character gestures, lyrical pacing.

13. Neo-Anime Intensity style:
    hard-edged shadows, neon city signage, extreme angle perspective, rapid insert cuts, high-contrast night palette, dramatic eye close-ups, kinetic camera sweeps, impact-frame flashes.

14. 3D Family Warmth style:
    soft global illumination, expressive face close-ups, clean depth-of-field, rich color harmony, bouncy camera easing, comedic timing beats, bright specular eyes, cozy bounce light.

15. Stop-Motion Handmade style:
    tactile clay/felt textures, slight frame jitter, miniature set depth, practical tiny lights, handcrafted props, warm vignette, whimsical cut timing, charming imperfections.

16. Analog Horror style:
    CRT scanlines, crushed blacks, overexposed hotspots, found-footage shakiness, warning text overlays, audio dropouts, unsettling still frames, surveillance zoom, corrupted glitches.

17. Epic High Fantasy style:
    golden-hour god rays, sweeping crane reveals, wind-swept silhouettes, ornate costume detail, painterly haze, heroic wides, choral swell pacing, majestic slow push-ins.

18. Minimalist Arthouse Realism style:
    natural window light, long static takes, quiet room tone, muted neutrals, sparse blocking, observational distance, subtle handheld micro-movement, restrained cuts, candid imperfections.

WORKFLOW:
1. Identify the closest reference style(s) from the library.
2. Extract the signature cinematic soul.
3. Populate the cinematic_style object in your JSON output.
4. For styles not in the library, extrapolate: create a new bundle with consistent color + camera + texture + edit rhythm.
5. Let cinematic_style directives inform your shot descriptions, style_guide, and audio_design throughout the script.
"""
