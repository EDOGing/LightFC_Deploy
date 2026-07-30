package com.example.trackingapp.file

import android.content.res.AssetManager
import com.example.trackingapp.model.ModelInfo
import com.example.trackingapp.model.ModelRepository
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID

class BundledModelInstaller(
    private val assetManager: AssetManager,
    private val repository: ModelRepository,
) {
    fun ensureInstalled(): ModelInfo {
        val finalDirectory = File(repository.rootDirectory, MODEL_ID)
        if (isCurrent(finalDirectory)) return ModelInfo.fromDirectory(finalDirectory)

        val stagingDirectory = File(repository.rootDirectory, ".bundled-${UUID.randomUUID()}")
        check(stagingDirectory.mkdir()) { "无法创建内置模型临时目录" }
        try {
            SPECS.forEach { spec ->
                val destination = File(stagingDirectory, spec.name)
                assetManager.open("$ASSET_DIRECTORY/${spec.name}").use { input ->
                    FileOutputStream(destination).use { output -> input.copyTo(output) }
                }
                check(destination.length() == spec.size && sha256(destination) == spec.sha256) {
                    "内置模型校验失败：${spec.name}"
                }
            }

            val info = createInfo(stagingDirectory, finalDirectory)
            File(stagingDirectory, ModelInfo.METADATA_FILE_NAME)
                .writeText(info.toJson().toString(2), Charsets.UTF_8)

            if (finalDirectory.exists()) {
                check(finalDirectory.deleteRecursively()) { "无法替换旧版内置模型" }
            }
            check(stagingDirectory.renameTo(finalDirectory)) { "无法提交内置模型" }
            return ModelInfo.fromDirectory(finalDirectory)
        } catch (error: Throwable) {
            stagingDirectory.deleteRecursively()
            throw error
        }
    }

    private fun isCurrent(directory: File): Boolean {
        val info = runCatching { ModelInfo.fromDirectory(directory) }.getOrNull() ?: return false
        if (!info.isBundled || info.id != MODEL_ID) return false
        return SPECS.all { spec ->
            val file = File(directory, spec.name)
            file.isFile && file.length() == spec.size && sha256(file) == spec.sha256
        }
    }

    private fun createInfo(staging: File, finalDirectory: File) = ModelInfo(
        id = MODEL_ID,
        name = "LightFC",
        source = ModelInfo.SOURCE_BUNDLED,
        templateParamFileName = TEMPLATE_PARAM,
        templateBinFileName = TEMPLATE_BIN,
        trackingParamFileName = TRACKING_PARAM,
        trackingBinFileName = TRACKING_BIN,
        templateParamSize = File(staging, TEMPLATE_PARAM).length(),
        templateBinSize = File(staging, TEMPLATE_BIN).length(),
        trackingParamSize = File(staging, TRACKING_PARAM).length(),
        trackingBinSize = File(staging, TRACKING_BIN).length(),
        importedAtMillis = System.currentTimeMillis(),
        directory = finalDirectory,
    )

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02X".format(byte) }
    }

    private data class AssetSpec(val name: String, val size: Long, val sha256: String)

    companion object {
        private const val MODEL_ID = "builtin-lightfc"
        private const val ASSET_DIRECTORY = "models/lightfc"
        private const val TEMPLATE_PARAM = "lightfc_template.ncnn.param"
        private const val TEMPLATE_BIN = "lightfc_template.ncnn.bin"
        private const val TRACKING_PARAM = "lightfc_tracking.ncnn.param"
        private const val TRACKING_BIN = "lightfc_tracking.ncnn.bin"
        private val SPECS = listOf(
            AssetSpec(TEMPLATE_PARAM, 6_118L, "10429F8C0D80C0BD4786357E6C7FDA6279F1C99E0E23A9B7012D106CA35B9CDA"),
            AssetSpec(TEMPLATE_BIN, 2_138_012L, "28010FA155267B269FC23AD154EFFA3B0CCB63D11C27564414FEC64597FB1528"),
            AssetSpec(TRACKING_PARAM, 19_002L, "CC35DD47DAD620F1C22B2146A42C7AB9EDB62C4C4462FD6605CB4C4268168F23"),
            AssetSpec(TRACKING_BIN, 12_622_496L, "2660AC19C67FA12A17A212E38CCFD1F83FF491C471389AC80B92596C170E2F7E"),
        )
    }
}
