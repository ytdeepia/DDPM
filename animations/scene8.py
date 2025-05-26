from manim import *
from manim_voiceover import VoiceoverScene
import sys
import numpy as np

sys.path.append("../")


class Scene2_8(VoiceoverScene):
    def construct(self):
        # Libraries and dataset
        self.next_section(skip_animations=False)

        code = Code(
            code_file="../src/training_minimal.py",
            background="window",
            language="python",
            add_line_numbers=False,
            formatter_style=Code.get_styles_list()[15],
        )
        code.scale_to_fit_height(0.8 * config.frame_height).to_edge(
            LEFT, buff=0.1
        ).to_edge(UP, buff=0.1)

        torch_logo = SVGMobject("./img/pytorch_logo.svg", z_index=1, height=2)
        deepinv_logo = (
            ImageMobject("./img/deepinv_logo.png", z_index=1)
            .scale_to_fit_width(4)
            .next_to(torch_logo, RIGHT, buff=2.0, aligned_edge=DOWN)
        )

        Group(deepinv_logo, torch_logo).next_to(code, RIGHT, buff=1.0)

        rect_logo = SurroundingRectangle(
            deepinv_logo,
            color=WHITE,
            fill_opacity=1,
            buff=0.2,
            z_index=-1,
            corner_radius=0.2,
        )

        self.play(Create(code.background))

        self.play(Write(code.code_lines[0:3]))

        self.play(FadeIn(torch_logo))

        self.play(Indicate(code.code_lines[1], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[1], color=WHITE, scale_factor=1.1))

        self.play(FadeIn(deepinv_logo, rect_logo))

        self.play(FadeOut(deepinv_logo, rect_logo, torch_logo))

        brace_data = Brace(code.code_lines[4:20], RIGHT, buff=0.3, stroke_width=1)
        label_data = Tex("Setup the dataset", font_size=24).next_to(
            brace_data.get_tip(), RIGHT, buff=0.3
        )

        self.play(Write(code.code_lines[3:20]))
        self.play(
            LaggedStart(GrowFromCenter(brace_data), Write(label_data), lag_ratio=0.5)
        )

        dataset_name = (
            Tex("MNIST", font_size=32)
            .shift(config.frame_width / 4 * RIGHT)
            .to_edge(UP, buff=0.75)
        )
        dataset_name_ul = Underline(dataset_name)

        mnist_0 = (
            ImageMobject("./img/mnist_0.png", z_index=1)
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(1)
        )
        mnist_0.add(SurroundingRectangle(mnist_0, color=WHITE, buff=0.2))
        mnist_1 = (
            ImageMobject("./img/mnist_1.png", z_index=1)
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(1)
        )
        mnist_1.add(SurroundingRectangle(mnist_1, color=WHITE, buff=0.2))
        mnist_2 = (
            ImageMobject("./img/mnist_2.png", z_index=1)
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(1)
        )
        mnist_2.add(SurroundingRectangle(mnist_2, color=WHITE, buff=0.2))
        mnist_3 = (
            ImageMobject("./img/mnist_3.png", z_index=1)
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(1)
        )
        mnist_3.add(SurroundingRectangle(mnist_3, color=WHITE, buff=0.2))

        Group(mnist_0, mnist_1, mnist_2, mnist_3).arrange_in_grid(
            2, 2, buff=0.5
        ).next_to(dataset_name, DOWN, buff=0.75)

        self.play(
            LaggedStart(Write(dataset_name), Create(dataset_name_ul), lag_ratio=0.7)
        )

        self.play(FadeIn(mnist_0, mnist_1, mnist_2, mnist_3))

        brace_width = Brace(mnist_0, UP, buff=0.1, stroke_width=1)
        label_width = Tex("32", font_size=28).next_to(
            brace_width.get_tip(), UP, buff=0.1
        )

        brace_height = Brace(mnist_0, LEFT, buff=0.1, stroke_width=1)
        label_height = Tex("32", font_size=28).next_to(
            brace_height.get_tip(), LEFT, buff=0.1
        )

        self.play(
            LaggedStart(GrowFromCenter(brace_width), Write(label_width), lag_ratio=0.5),
            LaggedStart(
                GrowFromCenter(brace_height), Write(label_height), lag_ratio=0.5
            ),
        )

        self.play(
            FadeOut(
                brace_width,
                label_width,
                brace_height,
                label_height,
                dataset_name_ul,
                dataset_name,
                mnist_0,
                mnist_1,
                mnist_2,
                mnist_3,
                brace_data,
                label_data,
            )
        )

        # Model, inputs and diffusion constants
        self.next_section(skip_animations=False)

        self.play(Write(code.code_lines[20:30]))

        brace_model = Brace(code.code_lines[20:30], RIGHT, buff=0.3, stroke_width=1)
        label_model = Tex("Setup the model", font_size=24).next_to(
            brace_model.get_tip(), RIGHT, buff=0.3
        )

        self.play(
            LaggedStart(GrowFromCenter(brace_model), Write(label_model), lag_ratio=0.5)
        )

        self.wait(0.64)

        # Create a DiffUNet diagram
        nn_title = (
            Text("DiffUNet", font_size=32)
            .shift(config.frame_width / 4 * RIGHT)
            .to_edge(UP, buff=0.75)
        )
        nn_title_ul = Underline(nn_title)

        nn_box_width = 0.3
        nn_box_height = 0.8
        nn_buff = 0.15

        input_box = RoundedRectangle(
            width=nn_box_width,
            height=nn_box_height,
            corner_radius=0.1,
            fill_color=BLUE_E,
            fill_opacity=0.6,
            stroke_width=1,
        )
        encoder1 = RoundedRectangle(
            width=nn_box_width,
            height=nn_box_height,
            corner_radius=0.1,
            fill_color=BLUE_D,
            fill_opacity=0.6,
            stroke_width=1,
        )
        encoder2 = RoundedRectangle(
            width=nn_box_width * 0.8,
            height=nn_box_height * 0.8,
            corner_radius=0.1,
            fill_color=BLUE_C,
            fill_opacity=0.6,
            stroke_width=1,
        )
        bottleneck = RoundedRectangle(
            width=nn_box_width * 0.6,
            height=nn_box_height * 0.6,
            corner_radius=0.1,
            fill_color=BLUE_B,
            fill_opacity=0.6,
            stroke_width=1,
        )
        decoder2 = RoundedRectangle(
            width=nn_box_width * 0.8,
            height=nn_box_height * 0.8,
            corner_radius=0.1,
            fill_color=BLUE_C,
            fill_opacity=0.6,
            stroke_width=1,
        )
        decoder1 = RoundedRectangle(
            width=nn_box_width,
            height=nn_box_height,
            corner_radius=0.1,
            fill_color=BLUE_D,
            fill_opacity=0.6,
            stroke_width=1,
        )
        output_box = RoundedRectangle(
            width=nn_box_width,
            height=nn_box_height,
            corner_radius=0.1,
            fill_color=BLUE_E,
            fill_opacity=0.6,
            stroke_width=1,
        )

        nn_boxes = (
            VGroup(
                input_box,
                encoder1,
                encoder2,
                bottleneck,
                decoder2,
                decoder1,
                output_box,
            )
            .arrange(RIGHT, buff=nn_buff)
            .shift(config.frame_width / 4 * RIGHT)
            .next_to(nn_title, DOWN, buff=0.75)
        )

        bounding_box = SurroundingRectangle(
            nn_boxes,
            color=WHITE,
            buff=0.2,
            z_index=-1,
            corner_radius=0.1,
        )

        arrows = VGroup()
        for i in range(len(nn_boxes) - 1):
            arrow = Arrow(
                nn_boxes[i].get_right(),
                nn_boxes[i + 1].get_left(),
                buff=0.0,
                color=WHITE,
            )
            arrows.add(arrow)

        nn_diagram = VGroup(nn_boxes, arrows, bounding_box).next_to(
            nn_title, DOWN, buff=0.75
        )

        self.play(Indicate(code.code_lines[24:27], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[24:27], color=WHITE, scale_factor=1.1))

        self.play(
            LaggedStart(Write(nn_title), Create(nn_title_ul), lag_ratio=0.5),
            FadeIn(nn_diagram),
        )

        self.play(Indicate(code.code_lines[27:28], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[27:28], color=WHITE, scale_factor=1.1))

        self.play(Indicate(code.code_lines[28:29], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[28:29], color=WHITE, scale_factor=1.1))

        self.play(
            FadeOut(
                nn_title,
                nn_title_ul,
                nn_diagram,
                brace_model,
                label_model,
            )
        )

        self.play(Write(code.code_lines[30:40]))

        brace_diffusion = Brace(code.code_lines[30:40], RIGHT, buff=1.0, stroke_width=1)
        label_diffusion = Tex("Diffusion constants", font_size=24).next_to(
            brace_diffusion.get_tip(), RIGHT, buff=0.3
        )

        self.play(
            LaggedStart(
                GrowFromCenter(brace_diffusion),
                Write(label_diffusion),
                lag_ratio=0.5,
            )
        )

        beta_axes = Axes(
            x_range=[0, 1000, 200],
            y_range=[0, 0.02, 0.005],
            axis_config={"include_tip": False, "include_numbers": True},
            x_length=4,
            y_length=3,
            x_axis_config={
                "numbers_to_include": [0, 200, 400, 600, 800, 1000],
                "font_size": 14,
            },
            y_axis_config={"numbers_to_include": [0, 0.01, 0.02], "font_size": 14},
        ).shift(config.frame_width / 4 * RIGHT)

        beta_x_label = Text("Timesteps", font_size=14).next_to(
            beta_axes.x_axis, DOWN, buff=0.1
        )
        beta_y_label = (
            Text("Beta", font_size=14)
            .next_to(beta_axes.y_axis, LEFT, buff=0.1)
            .rotate(PI / 2)
        )

        start_beta = 1e-4
        end_beta = 0.02

        beta_graph = beta_axes.plot(
            lambda x: start_beta + (end_beta - start_beta) * x / 1000,
            x_range=[0, 1000],
            color=BLUE,
            stroke_width=2,
        )

        beta_start_point = Dot(beta_axes.coords_to_point(0, start_beta), color=RED)
        beta_end_point = Dot(beta_axes.coords_to_point(1000, end_beta), color=RED)

        plot = VGroup(
            beta_axes,
            beta_x_label,
            beta_y_label,
            beta_graph,
            beta_start_point,
            beta_end_point,
        ).to_edge(RIGHT, buff=0.75)

        self.play(
            LaggedStart(
                FadeIn(beta_axes, beta_x_label, beta_y_label),
                lag_ratio=0.5,
            )
        )

        self.play(
            LaggedStart(
                FadeIn(beta_start_point),
                Create(beta_graph),
                FadeIn(beta_end_point),
                lag_ratio=0.5,
            ),
            run_time=1.5,
        )

        beta_cosine_graph = beta_axes.plot(
            lambda x: (
                0.5 * (1 - np.cos(np.pi * x / 1000)) * (end_beta - start_beta)
                + start_beta
            ),
            x_range=[0, 1000],
            color=BLUE,
            stroke_width=2,
        )

        self.play(ReplacementTransform(beta_graph, beta_cosine_graph))
        self.play(ReplacementTransform(beta_cosine_graph, beta_graph))

        self.play(Indicate(code.code_lines[35:36], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[35:36], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[36:37], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[36:37], color=WHITE, scale_factor=1.1))

        self.play(
            FadeOut(
                beta_axes,
                beta_x_label,
                beta_y_label,
                beta_graph,
                beta_start_point,
                beta_end_point,
                brace_diffusion,
                label_diffusion,
            )
        )

        # Training loop
        self.next_section(skip_animations=False)

        self.play(Write(code.code_lines[40:59]))

        brace_training = Brace(code.code_lines[40:59], RIGHT, buff=0.8, stroke_width=1)
        label_training = Tex("Training loop", font_size=24).next_to(
            brace_training.get_tip(), RIGHT, buff=0.3
        )

        self.play(
            LaggedStart(
                GrowFromCenter(brace_training), Write(label_training), lag_ratio=0.5
            )
        )

        self.play(Indicate(code.code_lines[44:45], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[44:45], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[45:46], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[45:46], color=WHITE, scale_factor=1.1))

        clean_img = (
            ImageMobject("./img/mnist_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(1)
        )
        clean_img.add(SurroundingRectangle(clean_img, color=WHITE, buff=0.0))
        clean_img_label = (
            Tex("Clean image", font_size=24)
            .next_to(clean_img, UP, buff=0.1)
            .set_color(WHITE)
        )

        noised_img = (
            ImageMobject("./img/mnist_0_noised.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale_to_fit_width(1)
            .next_to(clean_img, RIGHT, buff=3.0)
        )
        noised_img.add(SurroundingRectangle(noised_img, color=WHITE, buff=0.0))
        noised_img_label = (
            Tex("Noised image", font_size=24)
            .next_to(noised_img, UP, buff=0.1)
            .set_color(WHITE)
        )

        arrow = Arrow(
            clean_img.get_right(),
            noised_img.get_left(),
            buff=0.1,
            color=WHITE,
        )

        formula = MathTex(
            r"\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon",
            font_size=24,
        ).next_to(arrow, DOWN, buff=0.4)

        diagram = Group(
            clean_img,
            clean_img_label,
            noised_img,
            noised_img_label,
            arrow,
            formula,
        ).to_edge(RIGHT, buff=0.5)

        self.play(FadeIn(diagram))

        self.play(Indicate(code.code_lines[47:52], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[47:52], color=WHITE), scale_factor=1.1)

        # Complete loss formula
        loss = MathTex(
            r"\mathbb{E}_{q, t} \left[ \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right]",
            font_size=24,
        ).move_to(config.frame_width / 4 * RIGHT + 0.5 * UP)

        loss_simple = MathTex(
            r"\mathbb{E}_{q, t} \left[ \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right]",
            font_size=24,
        ).move_to(config.frame_width / 4 * RIGHT + 0.5 * UP)
        self.play(FadeOut(diagram), run_time=0.9)

        self.play(Write(loss))

        self.play(Indicate(code.code_lines[54:55], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[54:55], color=WHITE, scale_factor=1.1))

        self.play(Transform(loss, loss_simple))

        self.play(FadeOut(loss), run_time=0.8)

        self.play(Write(code.code_lines[58:63]))

        # Training progress
        self.next_section(skip_animations=False)

        losses = np.load(f"../src/losses/losses_epoch_100.npy")

        # Create a smooth line plot of the losses
        loss_axes = (
            Axes(
                x_range=[0, len(losses), len(losses) // 10],
                y_range=[np.min(losses) - 0.1, np.max(losses) + 0.1, 0.5],
                axis_config={"include_tip": False},
                x_length=6,
                y_length=3.5,
                x_axis_config={"font_size": 20},
                y_axis_config={"font_size": 20},
            )
            .move_to(config.frame_width / 4 * RIGHT)
            .to_edge(UP, buff=0.5)
        )

        loss_x_label = Tex("Training iterations", font_size=24).next_to(
            loss_axes.x_axis, DOWN, buff=0.3
        )
        loss_y_label = (
            Tex("MSE", font_size=24)
            .rotate(PI / 2)
            .next_to(loss_axes.y_axis, LEFT, buff=0.1)
        )

        loss_graph = loss_axes.plot_line_graph(
            x_values=list(range(len(losses))),
            y_values=losses.tolist(),
            line_color=BLUE,
            stroke_width=2,
            add_vertex_dots=False,
        )
        plot = (
            VGroup(
                loss_axes,
                loss_x_label,
                loss_y_label,
                loss_graph["line_graph"],
            )
            .scale(0.8)
            .next_to(code, RIGHT, buff=2.0)
            .to_edge(UP, buff=0.25)
        )

        self.play(FadeOut(brace_training, label_training))
        self.play(FadeIn(loss_axes, loss_x_label, loss_y_label))
        self.play(Create(loss_graph["line_graph"]), run_time=3)

        def get_image(path):
            img = (
                ImageMobject(path, z_index=0)
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale_to_fit_width(0.75)
            )
            img.add(
                SurroundingRectangle(
                    img, color=WHITE, buff=0.0, z_index=1, stroke_width=1
                )
            )
            return img

        image_types = ["original", "noisy", "denoised"]
        timesteps = [50, 100, 400, 700]
        digits = [0, 1, 2]
        epochs = range(10, 110, 10)

        images = {}

        # Function to organize images for an epoch
        def create_image_grid(epoch, first_time=False):
            base_path = f"../src/img/val/epoch_{epoch}/"
            local_images = {}

            # Load all images
            for digit in digits:
                local_images[f"original_{digit}"] = get_image(
                    f"{base_path}/original/image_{digit}.png"
                )

                for timestep in timesteps:
                    local_images[f"noisy_{digit}_{timestep}"] = get_image(
                        f"{base_path}/{timestep}/noisy/image_{digit}.png"
                    )
                    local_images[f"denoised_{digit}_{timestep}"] = get_image(
                        f"{base_path}/{timestep}/denoised/image_{digit}.png"
                    )

            # Create columns
            columns = {}

            # Original column
            col_ori = Group()
            for digit in digits:
                col_ori.add(local_images[f"original_{digit}"])

            col_ori.arrange(DOWN, buff=0.2)
            columns["original"] = col_ori

            # Create columns for each timestep
            for timestep in timesteps:
                col = Group()

                for digit in digits:
                    img_pair = Group(
                        local_images[f"noisy_{digit}_{timestep}"],
                        local_images[f"denoised_{digit}_{timestep}"],
                    ).arrange(RIGHT, buff=0.15)
                    col.add(img_pair)

                col.arrange(DOWN, buff=0.2)
                columns[f"t{timestep}"] = col

            # Arrange columns side by side
            img_grid = Group(columns["original"])
            for timestep in timesteps:
                columns[f"t{timestep}"].next_to(img_grid, RIGHT, buff=0.3)
                img_grid.add(columns[f"t{timestep}"])

            img_grid.next_to(plot, DOWN, buff=0.25)

            # Create labels
            labels = Group()
            labels.add(
                Tex("Original", font_size=24)
                .next_to(columns["original"], DOWN, buff=0.1)
                .set_color(WHITE)
            )

            for timestep in timesteps:
                labels.add(
                    MathTex(f"t={timestep}", font_size=24).next_to(
                        columns[f"t{timestep}"], DOWN, buff=0.1
                    )
                )

            return img_grid, labels, local_images

        # Process the first epoch separately
        first_epoch = 10
        img_grid, labels, first_images = create_image_grid(first_epoch, first_time=True)

        # Calculate position on the loss plot for first epoch
        plot_x_pos = len(losses) * (first_epoch / 100)
        plot_y_pos = np.interp(plot_x_pos, range(len(losses)), losses)

        # Create dot for first epoch
        epoch_dot = Dot(
            loss_axes.coords_to_point(plot_x_pos, plot_y_pos),
            color=RED,
            radius=0.08,
            z_index=3,
        )

        # Show epoch counter for first epoch
        epoch_counter = Tex(
            f"Epoch: {first_epoch}/100", font_size=32, color=WHITE
        ).to_corner(UR, buff=0.5)

        self.play(FadeIn(img_grid, *labels), FadeIn(epoch_dot), FadeIn(epoch_counter))

        # Process remaining epochs
        for epoch in [e for e in epochs if e > first_epoch]:
            new_img_grid, new_labels, _ = create_image_grid(epoch)

            # Replace previous grid with new one
            self.remove(img_grid)
            self.add(new_img_grid)
            img_grid = new_img_grid

            # Calculate position on loss plot for this epoch
            plot_x_pos = len(losses) * (epoch / 100)
            plot_y_pos = np.interp(plot_x_pos, range(len(losses)), losses)

            # Update dot position
            new_epoch_dot = Dot(
                loss_axes.coords_to_point(plot_x_pos, plot_y_pos),
                color=RED,
                radius=0.08,
                z_index=3,
            )
            self.remove(epoch_dot)
            self.add(new_epoch_dot)
            epoch_dot = new_epoch_dot

            # Update epoch counter
            self.play(
                Transform(
                    epoch_counter,
                    Tex(f"Epoch: {epoch}/100", font_size=32, color=WHITE).to_corner(
                        UR, buff=0.5
                    ),
                )
            )
            self.wait(0.5)

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))
        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_8()
    scene.render()
