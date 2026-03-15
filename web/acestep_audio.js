import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "AcestepCPP.AudioPreview",

    nodeCreated(node) {
        if (node.comfyClass !== "AcestepCPPAudioPlayer") return;

        node.onExecuted = function (output) {
            const audios = output?.audio;

            // Remove audio widgets from a previous run.
            this.widgets = (this.widgets ?? []).filter(w => !w._aceAudio);

            if (!audios?.length) return;

            for (const a of audios) {
                const src = api.apiURL(
                    `/view?filename=${encodeURIComponent(a.filename)}`
                    + `&subfolder=${encodeURIComponent(a.subfolder ?? "")}`
                    + `&type=${encodeURIComponent(a.type)}`
                );

                const el = document.createElement("audio");
                el.src = src;
                el.controls = true;
                el.style.cssText = "width:100%; margin-top:4px;";

                const w = this.addDOMWidget("audio_preview", "preview", el, {
                    serialize: false,
                    hideOnZoom: false,
                });
                w._aceAudio = true;
            }

            this.setSize([this.size[0], this.computeSize()[1]]);
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
