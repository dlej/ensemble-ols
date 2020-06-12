from manimlib.imports import *

from matplotlib import cm, colors, pyplot as plt

################################################################################
# constants
################################################################################

TITLE_SCALE = 1.4
SMALL_SCALE = 0.7
PATH_TO_IMAGES = 'images'
MARGIN_SCALE = 0.9
CMAP = colors.LinearSegmentedColormap.from_list('3b1b_bwr', [BLUE, WHITE, RED])
TICK_WIDTH = 0.3
TICK_SCALE = SMALL_SCALE
COLOR_CYCLE = [BLUE, YELLOW, RED, GREEN]
TEXT_NOSPACE_BUFFER = 0.03

################################################################################
# useful things
################################################################################

class DataMatrix(Rectangle):

    def __init__(self, n_rows, n_cols, **kwargs):

        Rectangle.__init__(self, **kwargs)

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.patches = VGroup()
    
    def transform_matrix_coords(self, i, j):
        """Returns the top-left corner of matrix element i, j
        """
        return self.get_corner(UL) + (i / self.n_rows)*self.get_height()*DOWN + (j / self.n_cols)*self.get_width()*RIGHT

    def get_row_height(self):
        return self.get_height() / self.n_rows

    def get_col_width(self):
        return self.get_width() / self.n_cols

class Patch(VGroup):

    CONFIG = {
        'fill_opacity': 0.5
    }

    def __init__(self, data_matrix, row_groups, col_groups, is_collected=False, **kwargs):

        digest_config(self, self.CONFIG)
        for k, v in self.CONFIG.items():
            if k not in kwargs:
                kwargs[k] = v
        VGroup.__init__(self, **kwargs)
        
        self.data_matrix = data_matrix
        self.patch_objects = self.create_patch_objects(row_groups, col_groups, **kwargs)

        if not is_collected:
            self.collected_patch = self.create_collected_patch(row_groups, col_groups, **kwargs)
        else:
            self.collected_patch = None

        self.add(self.patch_objects)
    
    def create_patch_objects(self, row_groups, col_groups, **kwargs):

        patch_objects = VGroup()
        for i, n_rows in row_groups:
            for j, n_cols in col_groups:
                patch = self.create_patch(i, j, n_rows, n_cols, **kwargs)
                patch_objects.add(patch)
        
        return patch_objects

    def create_patch(self, i, j, n_rows, n_cols, **kwargs):

        patch = Rectangle(
            width=self.data_matrix.get_col_width() * n_cols,
            height=self.data_matrix.get_row_height() * n_rows,
            **kwargs
        )
        patch.move_to(self.data_matrix.transform_matrix_coords(i, j), aligned_edge=UL)
        return patch
    
    def create_collected_patch(self, row_groups, col_groups, **kwargs):
        
        collected_row_groups = []
        i = 0
        for _, n_rows in row_groups:
            collected_row_groups.append((i, n_rows))
            i += n_rows

        collected_col_groups = []
        j = 0
        for _, n_cols in col_groups:
            collected_col_groups.append((j, n_cols))
            j += n_cols

        return self.__class__(self.data_matrix, collected_row_groups, collected_col_groups, is_collected=True, **kwargs)
    
    @classmethod
    def from_indices(cls, data_matrix, rows, cols, **kwargs):

        rows_bool = np.zeros(data_matrix.n_rows, dtype=bool)
        rows_bool[rows] = True
        cols_bool = np.zeros(data_matrix.n_cols, dtype=bool)
        cols_bool[cols] = True

        return cls.from_boolean(data_matrix, rows_bool, cols_bool, **kwargs)
    
    @classmethod
    def from_boolean(cls, data_matrix, rows_bool, cols_bool, **kwargs):
        return cls(data_matrix, boolean_vector_to_groups(rows_bool), boolean_vector_to_groups(cols_bool), **kwargs)


def boolean_vector_to_groups(v):

    v = list(v)
    starts = [i for i, (x, y) in enumerate(zip([False] + v[:-1], v)) if y and not x]
    ends = [i + 1 for i, (x, y) in enumerate(zip(v, v[1:] + [False])) if x and not y]

    return [(start, end - start) for start, end in zip(starts, ends)]

def sample_rows_cols(n_rows, n_cols, p_rows=0.5, p_cols=0.5, require_more_rows=False):

    rows_bool = np.random.rand(n_rows) < p_rows
    cols_bool = np.random.rand(n_cols) < p_cols

    if require_more_rows and sum(rows_bool) < sum(cols_bool):
        return sample_rows_cols(n_rows, n_cols, p_rows=p_rows, p_cols=p_cols, require_more_rows=True)
    else:
        return rows_bool, cols_bool

################################################################################
# begin scenes
################################################################################

class TitleScene(Scene):

    def construct(self):

        title = TextMobject('The Implicit Regularization of \\\\ Ordinary Least Squares Ensembles')
        title.scale(TITLE_SCALE)
        subtitle = TextMobject('AISTATS 2020')
        title_vg = VGroup(title, subtitle)
        title_vg.arrange(direction=DOWN, center=True)

        self.play(Write(title_vg))
        self.wait()

        self.play(FadeOut(title_vg))
        self.wait()

class AuthorScene(Scene):

    def construct(self):

        authors = [
            ['Daniel LeJeune', 'daniel.jpg', 'Rice University'],
            ['Hamid Javadi', 'hamid.jpg', 'Rice University'],
            ['Richard G. Baraniuk', 'baraniuk.jpg', 'Rice University']
        ]

        photos = []
        author_affils = []

        for i, (author_name, photo_filename, university) in enumerate(authors):

            photo = ImageMobject(os.path.join(PATH_TO_IMAGES, photo_filename))
            photo.scale(MARGIN_SCALE)
            photo.move_to([-3, 2 - 2*i, 0], aligned_edge=RIGHT)
            photos.append(photo)

            name_mobject = TextMobject(author_name)
            univ_mobject = TextMobject(university)
            univ_mobject.scale(SMALL_SCALE)

            author_affil_vg = VGroup(name_mobject, univ_mobject)
            author_affil_vg.arrange(direction=DOWN, aligned_edge=LEFT)
            author_affil_vg.next_to(photo, direction=RIGHT)
            author_affils.append(author_affil_vg)

        self.play(
            LaggedStartMap(
                lambda x: LaggedStartMap(FadeIn, x, lag_ratio=0.1),
                Group(Group(*photos), VGroup(*author_affils)),
                lag_ratio=0.5
            )
        )
        self.wait()
        self.play(
            LaggedStartMap(
                lambda x: LaggedStartMap(FadeOut, x, lag_ratio=0.1),
                Group(*[Group(*z) for z in zip(photos, author_affils)]),
                lag_ratio=0.1
            )
        )
        self.wait()

class EnsembleIntroScene(Scene):

    def construct(self):

        ensemble = TextMobject('Ensemble methods')
        ensemble.scale(TITLE_SCALE)
        ensemble.shift(2*UP)
        self.play(Write(ensemble))
        self.wait()

        rf_boosting = TextMobject('e.g., random forests, boosting', substrings_to_isolate=['random forests,'])
        rf_boosting.next_to(ensemble, DOWN, buff=MED_LARGE_BUFF)
        rf = rf_boosting.get_part_by_tex('random forests,')
        self.play(Write(rf_boosting))
        self.wait()

        rs_nonadaptive = TextMobject('random sampling, non-adaptive', color=BLUE, substrings_to_isolate=['random sampling,'])
        rs_nonadaptive.next_to(rf, DOWN)
        rs = rs_nonadaptive.get_part_by_tex('random sampling,')
        nonadaptive = rs_nonadaptive.get_part_by_tex('non-adaptive')
        self.play(
            Write(rs)
        )
        self.wait()
        self.play(
            Write(nonadaptive)
        )
        self.wait()
        self.play(
            rf_boosting.set_color, GRAY,
            rf.set_color, BLUE
        )
        self.wait()

        trees_to_ols = TextMobject(r'decision trees $\to$ OLS', substrings_to_isolate=['decision trees', 'OLS'])
        trees = trees_to_ols.get_part_by_tex('decision trees')
        trees.set_color(BLUE)
        to = trees_to_ols.get_part_by_tex(r'$\to$')
        ols = trees_to_ols.get_part_by_tex('OLS')
        ols.set_color(YELLOW)
        trees_to_ols.next_to(rs_nonadaptive, DOWN, coor_mask=UP)
        self.play(Write(trees))
        self.wait()
        self.play(Write(VGroup(to, ols)))
        self.wait()
        self.play(
            ensemble.set_color, YELLOW, 
            rs_nonadaptive.set_color, YELLOW
        )
        self.wait()

        ridge = TextMobject(r'$\Rightarrow$ ridge regularization', color=YELLOW)
        ridge.next_to(trees_to_ols, DOWN)
        self.play(Write(ridge))
        self.wait()



class PatchScene(Scene):

    def construct(self):

        n_rows, n_cols = 12, 15
        matrix_scale = 1 / 3

        X = DataMatrix(n_rows, n_cols, height=n_rows*matrix_scale, width=n_cols*matrix_scale)
        X_title = TextMobject(r'$\mathbf{X}$')
        X_title.next_to(X, UP)

        self.play(ShowCreation(X))
        self.play(Write(X_title))
        self.wait()

        equals = TextMobject(r'$\approx$')
        equals.next_to(X, LEFT)

        y = DataMatrix(n_rows, 1, height=n_rows*matrix_scale, width=matrix_scale)
        y.next_to(equals, LEFT)
        y_title = TextMobject(r'$\mathbf{y}$')
        y_title.next_to(y, UP)

        beta = DataMatrix(n_cols, 1, height=n_cols*matrix_scale, width=matrix_scale)
        beta.next_to(X, RIGHT)
        beta_title = TextMobject(r'$\boldsymbol{\beta}$')
        beta_title.next_to(beta, UP)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    ShowCreation(y),
                    Write(y_title),
                    lag_ratio=0.5
                ),
                Write(equals),
                AnimationGroup(
                    ShowCreation(beta),
                    Write(beta_title),
                    lag_ratio=0.5
                ),
                lag_ratio=1
            )
        )
        self.wait()

        yXbeta = VGroup(
            y, y_title,
            equals,
            X, X_title,
            beta, beta_title
        )

        self.play(
            ApplyFunction(
                lambda x: x.scale(0.7).shift(UP),
                yXbeta
            )
        )

        row_groups = [(1, 2), (4, 2), (8, 1), (10, 2)]
        col_groups = [(1, 1), (4, 1), (6, 3), (14, 1)]

        y_mp = Patch(y, row_groups, [(0, 1)], color=RED)
        X_rows_mp = Patch(X, row_groups, [(0, n_cols)], color=RED)

        y_sample_T = TextMobject(r'$\mathbf{T}^\top$', color=RED)
        y_sample_T.scale(0.7)
        y_sample_T.next_to(y_title, LEFT, buff=TEXT_NOSPACE_BUFFER, aligned_edge=DOWN)
        y_sample_T.shift(0.06*UP)
        
        X_sample_T = y_sample_T.copy()
        X_sample_T.next_to(X_title, LEFT, buff=TEXT_NOSPACE_BUFFER, aligned_edge=DOWN)
        
        sample_mats_T = VGroup(y_sample_T, X_sample_T)

        self.play(
            FadeIn(y_mp),
            FadeIn(X_rows_mp),
            FadeIn(sample_mats_T)
        )
        self.wait()
        self.play(
            FadeOut(y_mp),
            FadeOut(X_rows_mp),
            FadeOut(sample_mats_T)
        )
        self.wait()

        beta_mp = Patch(beta, col_groups, [(0, 1)], color=GREEN)
        X_cols_mp = Patch(X, [(0, n_rows)], col_groups, color=GREEN)

        X_sample_S = TextMobject(r'$\mathbf{S}$', color=GREEN)
        X_sample_S.scale(0.7)
        X_sample_S.next_to(X_title, RIGHT, buff=TEXT_NOSPACE_BUFFER, aligned_edge=DOWN)

        beta_sample_S = TextMobject(r'$\mathbf{S}^\top$', color=GREEN)
        beta_sample_S.scale(0.7)
        beta_sample_S.next_to(beta_title, LEFT, buff=TEXT_NOSPACE_BUFFER, aligned_edge=DOWN)
        beta_sample_S.shift(0.06*UP)

        sample_mats_S = VGroup(X_sample_S, beta_sample_S)

        self.play(
            FadeIn(beta_mp),
            FadeIn(X_cols_mp),
            FadeIn(sample_mats_S)
        )
        self.wait()
        self.play(
            FadeOut(beta_mp),
            FadeOut(X_cols_mp),
            FadeOut(sample_mats_S)
        )
        self.wait()

        for t in range(5):

            if t == 0:
                X_mp = Patch(X, row_groups, col_groups, color=BLUE)

            else:
                rows_bool, cols_bool = sample_rows_cols(n_rows=n_rows, n_cols=n_cols, p_rows=7/12, p_cols=6/15, require_more_rows=True)
                y_mp = Patch.from_boolean(y, rows_bool, np.ones(1, dtype=bool), color=RED)
                X_mp = Patch.from_boolean(X, rows_bool, cols_bool, color=BLUE)
                beta_mp = Patch.from_boolean(beta, cols_bool, np.ones(1, dtype=bool), color=GREEN)

            yXbeta_mp = VGroup(y_mp, X_mp, beta_mp)
            collected_yXbeta_mp = VGroup(*(m.collected_patch for m in yXbeta_mp))
            collected_yXbeta_mp.arrange()
            collected_yXbeta_mp.scale(0.5)
            collected_yXbeta_mp.move_to(5*LEFT + 2*DOWN + 2*t*RIGHT)

            y_sample_T = TextMobject(r'$\mathbf{T}_{%d}^\top$' % (t + 1), color=RED)
            y_sample_T.scale(0.7)
            y_sample_T.next_to(y_title, LEFT, buff=TEXT_NOSPACE_BUFFER, aligned_edge=DOWN)
            
            X_sample_T = y_sample_T.copy()
            X_sample_T.next_to(X_title, LEFT, buff=TEXT_NOSPACE_BUFFER, aligned_edge=DOWN)
            X_sample_T.shift(0.087*DOWN)
            X_sample_S = TextMobject(r'$\mathbf{S}_{%d}$' % (t + 1), color=GREEN)
            X_sample_S.scale(0.7)
            X_sample_S.next_to(X_title, RIGHT, buff=TEXT_NOSPACE_BUFFER, aligned_edge=DOWN)
            X_sample_S.shift(0.05*DOWN)

            beta_sample_S = TextMobject(r'$\mathbf{S}_{%d}^\top$' % (t + 1), color=GREEN)
            beta_sample_S.scale(0.7)
            beta_sample_S.next_to(beta_title, LEFT, buff=TEXT_NOSPACE_BUFFER, aligned_edge=DOWN)
            beta_sample_S.shift(0.02*DOWN)

            sample_mats = VGroup(y_sample_T, X_sample_T, X_sample_S, beta_sample_S)

            self.play(
                FadeIn(yXbeta_mp),
                FadeIn(sample_mats)
            )
            if t == 0:
                self.wait()
            self.play(
                Transform(yXbeta_mp, collected_yXbeta_mp),
                FadeOut(sample_mats)
            )
            if t == 0:
                self.wait()
        
        self.wait()

        ldots = TextMobject(r'\ldots')
        ldots.scale(3)
        ldots.move_to(5*RIGHT + 2*DOWN)
        self.play(FadeIn(ldots))
        self.wait()

class FormalIntroScene(Scene):

    def construct(self):

        X = TextMobject(r'$\mathbf{X} \in \mathbb{R}^{n \times p}$')
        beta = TextMobject(r'$\boldsymbol{\beta} \in \mathbb{R}^p$')
        X_beta_vg = VGroup(X, beta)
        X_beta_vg.arrange(RIGHT, buff=MED_LARGE_BUFF)
        X_beta_vg.shift(2*UP)
        self.play(Write(X))
        self.wait()
        self.play(Write(beta))
        self.wait()

        y = TextMobject(r'$\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \sigma \mathbf{z}$')
        z = TextMobject(r'$\mathbf{z} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$')
        y_z_vg = VGroup(y, z)
        y_z_vg.arrange(RIGHT, buff=MED_LARGE_BUFF)
        y_z_vg.next_to(X_beta_vg, DOWN)
        self.play(Transform(X_beta_vg.copy(), y))
        self.wait()
        self.play(Write(z))
        self.wait()

        beta_ens = TextMobject(r'''$
            \widehat{\boldsymbol{\beta}}^{\mathrm{ens}}
            = \frac{1}{k} \sum_{i=1}^k \widehat{\boldsymbol{\beta}}^{(i)}
        $''')
        beta_hat = TextMobject(r'''$
            \widehat{\boldsymbol{\beta}}^{(i)}
            = \underset{\boldsymbol{\beta}'}{\mathrm{arg min}} \,
            ||\mathbf{T}_i^\top (\mathbf{X} \mathbf{S}_i \mathbf{S}_i^\top \boldsymbol{\beta}' - \mathbf{y})||_2^2
        $''')
        beta_ens.next_to(y_z_vg, DOWN)
        beta_hat.next_to(beta_ens, DOWN)
        self.play(Write(beta_ens))
        self.wait()
        self.play(Write(beta_hat))
        self.wait()

        dim_sampled = TextMobject(r'''$
            \mathbb{E}_{\mathbf{T}, \mathbf{S}} [
                \mathrm{dim}(\mathbf{T}^\top \mathbf{X} \mathbf{S})
            ] = (\eta n, \alpha p)
        $''')
        dim_sampled.next_to(beta_hat, DOWN)
        self.play(Write(dim_sampled))
        self.wait()

class RiskPairwiseScene(Scene):

    def construct(self):
        # TODO
        pass

class AcknowledgmentScene(Scene):

    def construct(self):

        ack_text = '''
        This work was supported by NSF grants CCF-1911094, IIS-1838177, and IIS1730574; 
        ONR grants N00014-18-12571 and N00014- 17-1-2551; AFOSR grant FA9550-18-1-0478; 
        DARPA grant G001534-7500; and a Vannevar Bush Faculty Fellowship, ONR grant N00014-18-1-2047.
        '''
        ack = TextMobject(ack_text)

        self.add(ack)