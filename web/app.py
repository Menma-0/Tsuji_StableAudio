"""
Gradio Web UI for Onomatopoeia-based Audio Editing

Features:
- Chain editing: up to 3 sequential edits on the same audio
- Reference audio: listen to target sounds before editing
- Participant ID and data saving functionality
"""
import gradio as gr
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
import sys
import json
import shutil
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from inference.pipeline import InferencePipeline
from config import default_config

# Global pipeline instance
pipeline = None

# Output directory for saved data
OUTPUT_DIR = Path(__file__).parent.parent / "experiment_data"


def load_pipeline():
    """Load the inference pipeline."""
    global pipeline
    if pipeline is None:
        print("Loading models...")
        pipeline = InferencePipeline()
        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "experiment_38dim_v2" / "best.pt"
        if checkpoint_path.exists():
            pipeline.load_models(checkpoint_path)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return pipeline


def edit_audio_step(
    input_audio,
    source_onomatopoeia,
    target_onomatopoeia,
    alpha,
    previous_result,
    step_num,
):
    """Execute one editing step."""
    try:
        pipe = load_pipeline()

        # Determine input audio
        if step_num == 1:
            if input_audio is None:
                return None, "音声ファイルをアップロードしてください", source_onomatopoeia
            audio_path = input_audio
        else:
            if previous_result is None:
                return None, f"先に編集{step_num - 1}を実行してください", source_onomatopoeia
            audio_path = previous_result

        # Validate inputs
        if not source_onomatopoeia or not source_onomatopoeia.strip():
            return None, "元のオノマトペを入力してください", source_onomatopoeia
        if not target_onomatopoeia or not target_onomatopoeia.strip():
            return None, "目標のオノマトペを入力してください", source_onomatopoeia

        # Create temp file for output
        output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        # Execute editing
        pipe.edit_audio(
            audio_path=audio_path,
            source_onomatopoeia=source_onomatopoeia.strip(),
            target_onomatopoeia=target_onomatopoeia.strip(),
            alpha=alpha,
            output_path=output_file.name,
        )

        status = f"完了: {source_onomatopoeia} → {target_onomatopoeia} (α={alpha})"
        return output_file.name, status, target_onomatopoeia

    except Exception as e:
        return None, f"エラー: {str(e)}", source_onomatopoeia


def edit_step1(input_audio, source_ono, target_ono, alpha):
    """Execute edit step 1."""
    result_path, status, next_source = edit_audio_step(
        input_audio, source_ono, target_ono, alpha, None, 1
    )
    # Show edit2 section if successful
    show_edit2 = gr.update(visible=True) if result_path else gr.update()
    # Return input_audio as comparison
    compare = input_audio if result_path else None
    return result_path, status, next_source, result_path, show_edit2, compare


def edit_step2(input_audio, source_ono, target_ono, alpha, prev_result):
    """Execute edit step 2."""
    result_path, status, next_source = edit_audio_step(
        input_audio, source_ono, target_ono, alpha, prev_result, 2
    )
    # Show edit3 section if successful
    show_edit3 = gr.update(visible=True) if result_path else gr.update()
    # Return prev_result (edit1 result) as comparison
    compare = prev_result if result_path else None
    return result_path, status, next_source, result_path, show_edit3, compare


def edit_step3(input_audio, source_ono, target_ono, alpha, prev_result):
    """Execute edit step 3."""
    result_path, status, next_source = edit_audio_step(
        input_audio, source_ono, target_ono, alpha, prev_result, 3
    )
    # Return prev_result (edit2 result) as comparison
    compare = prev_result if result_path else None
    return result_path, status, result_path, compare


def reset_all():
    """Reset all states (except participant ID)."""
    return (
        None,  # reference_audio
        None,  # input_audio
        None,  # result_audio1
        None,  # result_audio2
        None,  # result_audio3
        None,  # compare_audio1
        None,  # compare_audio2
        None,  # compare_audio3
        "",    # source_ono1
        "",    # target_ono1
        "",    # source_ono2
        "",    # target_ono2
        "",    # source_ono3
        "",    # target_ono3
        4.0,   # alpha1
        4.0,   # alpha2
        4.0,   # alpha3
        "",    # status1
        "",    # status2
        "",    # status3
        None,  # result1_state
        None,  # result2_state
        None,  # result3_state
        "",    # save_status
        gr.update(visible=False),  # edit2_section
        gr.update(visible=False),  # edit3_section
    )


def save_and_next(
    participant_id,
    reference_audio,
    input_audio,
    source_ono1, target_ono1, alpha1,
    source_ono2, target_ono2, alpha2,
    source_ono3, target_ono3, alpha3,
    result1_state,
    result2_state,
    result3_state,
):
    """Save experiment data and reset for next trial."""
    # Validate participant ID
    if not participant_id or not participant_id.strip():
        error_return = [gr.update()] * 23 + ["エラー: 参加者IDを入力してください"]
        error_return.extend([gr.update(), gr.update()])  # edit2, edit3 sections
        return tuple(error_return)

    # Check if at least one edit was performed
    if result1_state is None:
        error_return = [gr.update()] * 23 + ["エラー: 保存前に編集を実行してください"]
        error_return.extend([gr.update(), gr.update()])
        return tuple(error_return)

    try:
        # Create participant directory
        participant_dir = OUTPUT_DIR / participant_id.strip()
        participant_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for this trial
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_dir = participant_dir / f"trial_{timestamp}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Prepare metadata
        metadata = {
            "participant_id": participant_id.strip(),
            "timestamp": datetime.now().isoformat(),
            "reference_audio_path": str(reference_audio) if reference_audio else None,
            "input_audio_path": str(input_audio) if input_audio else None,
            "edits": [],
        }

        # Save edited audio files
        if result1_state:
            edit1_filename = f"edit1_{timestamp}.wav"
            shutil.copy2(result1_state, trial_dir / edit1_filename)
            metadata["edits"].append({
                "step": 1,
                "source_onomatopoeia": source_ono1,
                "target_onomatopoeia": target_ono1,
                "alpha": alpha1,
                "output_file": edit1_filename,
            })

        if result2_state:
            edit2_filename = f"edit2_{timestamp}.wav"
            shutil.copy2(result2_state, trial_dir / edit2_filename)
            metadata["edits"].append({
                "step": 2,
                "source_onomatopoeia": source_ono2,
                "target_onomatopoeia": target_ono2,
                "alpha": alpha2,
                "output_file": edit2_filename,
            })

        if result3_state:
            edit3_filename = f"edit3_{timestamp}.wav"
            shutil.copy2(result3_state, trial_dir / edit3_filename)
            metadata["edits"].append({
                "step": 3,
                "source_onomatopoeia": source_ono3,
                "target_onomatopoeia": target_ono3,
                "alpha": alpha3,
                "output_file": edit3_filename,
            })

        # Save metadata
        with open(trial_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        num_edits = len(metadata["edits"])
        save_message = f"保存完了！（{num_edits}件の編集を {trial_dir.name} に保存）"

        # Return reset values
        return (
            None, None,  # reference_audio, input_audio
            None, None, None,  # result_audio 1,2,3
            None, None, None,  # compare_audio 1,2,3
            "", "", "", "", "", "",  # source/target ono 1,2,3
            4.0, 4.0, 4.0,  # alpha 1,2,3
            "", "", "",  # status 1,2,3
            None, None, None,  # result states
            save_message,
            gr.update(visible=False),  # edit2_section
            gr.update(visible=False),  # edit3_section
        )

    except Exception as e:
        error_return = [gr.update()] * 23 + [f"保存エラー: {str(e)}"]
        error_return.extend([gr.update(), gr.update()])
        return tuple(error_return)


def create_ui():
    """Create the Gradio interface."""

    css = """
    .compact-audio audio {
        height: 50px !important;
    }
    .compact-audio .wrap {
        padding: 5px !important;
    }
    #participant_id_box {
        max-width: 200px;
    }
    .edit-section {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    """

    with gr.Blocks(
        title="オノマトペ音声編集",
        theme=gr.themes.Soft(),
        css=css,
    ) as demo:

        # Header with title and participant ID
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# オノマトペ音声編集システム")
            with gr.Column(scale=1):
                participant_id = gr.Textbox(
                    label="参加者ID",
                    placeholder="ID入力",
                    elem_id="participant_id_box",
                    scale=1,
                )

        # State variables
        result1_state = gr.State(None)
        result2_state = gr.State(None)
        result3_state = gr.State(None)

        # Audio input section - side by side, compact
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 編集する音声")
                input_audio = gr.Audio(
                    label="",
                    type="filepath",
                    elem_classes="compact-audio",
                )
            with gr.Column(scale=1):
                gr.Markdown("### 参照音声（目標の音）")
                reference_audio = gr.Audio(
                    label="",
                    type="filepath",
                    elem_classes="compact-audio",
                )

        gr.Markdown("---")

        # Edit 1 - Always visible
        with gr.Group(elem_classes="edit-section"):
            gr.Markdown("### 編集 1")
            with gr.Row():
                with gr.Column(scale=1):
                    source_ono1 = gr.Textbox(
                        label="元のオノマトペ",
                        placeholder="例: コッ",
                        info="現在の音の印象",
                    )
                with gr.Column(scale=1):
                    target_ono1 = gr.Textbox(
                        label="目標のオノマトペ",
                        placeholder="例: ガシャン",
                        info="目指す音の印象",
                    )
                with gr.Column(scale=1):
                    alpha1 = gr.Slider(
                        minimum=0.1, maximum=5.0, value=4.0, step=0.1,
                        label="強度 (α)",
                        info="4.0=推奨",
                    )
            with gr.Row():
                edit_btn1 = gr.Button("編集1 実行", variant="primary")
                status1 = gr.Textbox(label="状態", interactive=False, scale=2)
            with gr.Row():
                with gr.Column(scale=1):
                    result_audio1 = gr.Audio(
                        label="編集1 結果（トリム可）",
                        interactive=True,
                        elem_classes="compact-audio",
                    )
                with gr.Column(scale=1):
                    compare_audio1 = gr.Audio(
                        label="編集前（比較用）",
                        interactive=False,
                        elem_classes="compact-audio",
                    )

        # Edit 2 - Initially hidden
        edit2_section = gr.Group(visible=False, elem_classes="edit-section")
        with edit2_section:
            gr.Markdown("### 編集 2（追加編集）")
            with gr.Row():
                with gr.Column(scale=1):
                    source_ono2 = gr.Textbox(
                        label="元のオノマトペ",
                        placeholder="（自動入力）",
                        info="編集1の結果から自動設定",
                    )
                with gr.Column(scale=1):
                    target_ono2 = gr.Textbox(
                        label="目標のオノマトペ",
                        placeholder="例: ドン",
                    )
                with gr.Column(scale=1):
                    alpha2 = gr.Slider(
                        minimum=0.1, maximum=5.0, value=4.0, step=0.1,
                        label="強度 (α)",
                        info="4.0=推奨",
                    )
            with gr.Row():
                edit_btn2 = gr.Button("編集2 実行", variant="primary")
                status2 = gr.Textbox(label="状態", interactive=False, scale=2)
            with gr.Row():
                with gr.Column(scale=1):
                    result_audio2 = gr.Audio(
                        label="編集2 結果（トリム可）",
                        interactive=True,
                        elem_classes="compact-audio",
                    )
                with gr.Column(scale=1):
                    compare_audio2 = gr.Audio(
                        label="編集1結果（比較用）",
                        interactive=False,
                        elem_classes="compact-audio",
                    )

        # Edit 3 - Initially hidden
        edit3_section = gr.Group(visible=False, elem_classes="edit-section")
        with edit3_section:
            gr.Markdown("### 編集 3（追加編集）")
            with gr.Row():
                with gr.Column(scale=1):
                    source_ono3 = gr.Textbox(
                        label="元のオノマトペ",
                        placeholder="（自動入力）",
                        info="編集2の結果から自動設定",
                    )
                with gr.Column(scale=1):
                    target_ono3 = gr.Textbox(
                        label="目標のオノマトペ",
                        placeholder="例: カーン",
                    )
                with gr.Column(scale=1):
                    alpha3 = gr.Slider(
                        minimum=0.1, maximum=5.0, value=4.0, step=0.1,
                        label="強度 (α)",
                        info="4.0=推奨",
                    )
            with gr.Row():
                edit_btn3 = gr.Button("編集3 実行", variant="primary")
                status3 = gr.Textbox(label="状態", interactive=False, scale=2)
            with gr.Row():
                with gr.Column(scale=1):
                    result_audio3 = gr.Audio(
                        label="編集3 結果（トリム可）",
                        interactive=True,
                        elem_classes="compact-audio",
                    )
                with gr.Column(scale=1):
                    compare_audio3 = gr.Audio(
                        label="編集2結果（比較用）",
                        interactive=False,
                        elem_classes="compact-audio",
                    )

        gr.Markdown("---")

        # Save status
        save_status = gr.Textbox(label="保存状態", interactive=False)

        # Action buttons
        with gr.Row():
            with gr.Column(scale=3):
                next_btn = gr.Button(
                    "保存して次へ",
                    variant="primary",
                    size="lg",
                )
            with gr.Column(scale=1):
                gr.Markdown("")  # spacer
            with gr.Column(scale=1):
                reset_btn = gr.Button(
                    "リセット",
                    variant="stop",
                    size="sm",
                )

        # Help text
        with gr.Accordion("使い方", open=False):
            gr.Markdown(
                """
                1. **参加者ID**を右上に入力
                2. **参照音声**（任意）: 目標となる音をアップロード
                3. **編集する音声**: 編集したい音声をアップロード
                4. **オノマトペ入力**: 元と目標のオノマトペを入力して「編集実行」
                5. 必要に応じて追加編集（編集2, 3）を実行
                6. **保存して次へ**で結果を保存

                **オノマトペ例**: コッ, ガシャン, ドン, カーン, パタパタ, ゴロゴロ
                """
            )

        # Event handlers
        edit_btn1.click(
            fn=edit_step1,
            inputs=[input_audio, source_ono1, target_ono1, alpha1],
            outputs=[result_audio1, status1, source_ono2, result1_state, edit2_section, compare_audio1],
        )

        edit_btn2.click(
            fn=edit_step2,
            inputs=[input_audio, source_ono2, target_ono2, alpha2, result1_state],
            outputs=[result_audio2, status2, source_ono3, result2_state, edit3_section, compare_audio2],
        )

        edit_btn3.click(
            fn=edit_step3,
            inputs=[input_audio, source_ono3, target_ono3, alpha3, result2_state],
            outputs=[result_audio3, status3, result3_state, compare_audio3],
        )

        reset_btn.click(
            fn=reset_all,
            inputs=[],
            outputs=[
                reference_audio, input_audio,
                result_audio1, result_audio2, result_audio3,
                compare_audio1, compare_audio2, compare_audio3,
                source_ono1, target_ono1,
                source_ono2, target_ono2,
                source_ono3, target_ono3,
                alpha1, alpha2, alpha3,
                status1, status2, status3,
                result1_state, result2_state, result3_state,
                save_status,
                edit2_section, edit3_section,
            ],
        )

        next_btn.click(
            fn=save_and_next,
            inputs=[
                participant_id,
                reference_audio, input_audio,
                source_ono1, target_ono1, alpha1,
                source_ono2, target_ono2, alpha2,
                source_ono3, target_ono3, alpha3,
                result1_state, result2_state, result3_state,
            ],
            outputs=[
                reference_audio, input_audio,
                result_audio1, result_audio2, result_audio3,
                compare_audio1, compare_audio2, compare_audio3,
                source_ono1, target_ono1,
                source_ono2, target_ono2,
                source_ono3, target_ono3,
                alpha1, alpha2, alpha3,
                status1, status2, status3,
                result1_state, result2_state, result3_state,
                save_status,
                edit2_section, edit3_section,
            ],
        )

    return demo


def main():
    """Main entry point."""
    print("オノマトペ音声編集システムを起動中...")
    print("モデル読み込み中（少々お待ちください）...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"データ保存先: {OUTPUT_DIR}")

    try:
        load_pipeline()
        print("モデル読み込み完了！")
    except Exception as e:
        print(f"警告: モデル読み込み失敗: {e}")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,
        share=False,
    )


if __name__ == "__main__":
    main()
