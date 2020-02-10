<!doctype html>
<html lang="ja">

<head>
    <!-- Required meta tags -->
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous" />
    <title>VividFace</title>
    <style>
        .foot.container {
            margin-top: 1em;
            margin-bottom: 2em;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>Vivid Face&nbsp;<small>-Neural Network-</small></h1>
        <hr />

        <%
            DETECT_COUNT = -1
            try:
                DETECT_COUNT = len(list_result)
            except:
                pass
            end
        %>

        % if DETECT_COUNT < 1:
            % if DETECT_COUNT == 0:
            <div class="alert alert-warning" role="alert">
                キャラクターの認識が行えませんでした。
            </div>
            % end
        % else:
        <div class="alert alert-light" role="alert">
            <div class="text-center">
                <img src="{{pathname}}" class="img-fluid img-thumbnail" />
            </div>
        </div>

        <div>
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>#</th>
                        <th class="text-center">Face</th>
                        <th class="text-center">Chara</th>
                        <th class="text-right">Corona</th>
                        <th class="text-right">Einhald</th>
                        <th class="text-right">Fuka</th>
                        <th class="text-right">Miura</th>
                        <th class="text-right">Rinne</th>
                        <th class="text-right">Rio</th>
                        <th class="text-right">Vivio</th>
                    </tr>
                </thead>
                <tbody>
                    % for idx, v in enumerate(list_result):
                    <tr>
                        <th>{{idx}}</th>
                        <th class="text-center"><img class="img-fluid img-thumbnail" src="{{v[0]}}" /></th>
                        <th class="text-center">{{v[1]}}</th>
                        % for rate in v[2]:
                        <td class="text-right"><small>{{"%.2f" % (rate * 100)}}%</small></td>
                        % end
                    </tr>
                    % end
                </tbody>
            </table>
        </div>
        <hr />
        % end

        <form action="./decide" method="post" enctype="multipart/form-data">
            <div class="alert alert-info" role="alert">
                ファイル名に日本語が含まれるとアップロードに失敗することがあります。
            </div>
            <div class="form-group">
                <label for="id_file">判定したい画像をアップロードしてください</label>
                <input id="id_file" type="file" class="form-control-file" name="upload" class="form-control-file" />
            </div>
            <button type="submit" class="btn btn-primary">判定してもらう！</button>
        </form>
    </div>

    <div class="ui foot right aligned container">
    </div>

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"
        integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN"
        crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"
        integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q"
        crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"
        integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl"
        crossorigin="anonymous"></script>
</body>

</html>